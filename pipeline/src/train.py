from __future__ import annotations
import os
import json
import time
import math
import pathlib
from typing import Dict, List, Tuple
from joblib import dump

import wandb

#import faiss

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

import torch, torch.nn.functional as F

from torch.utils.data import Dataset

#from sentence_transformers import SentenceTransformer

from src.towers import ItemTower #UserAux, info_nce_loss

ROOT   = pathlib.Path(__file__).parents[2]
ART    = ROOT / "artifacts"
DATA_P = ART / "items.parquet"
EV_P   = ART / "events.parquet"
USER_P = ART / "users.parquet"
MAP_P  = ART / "mappings.json"
TEXT_EMB_P = ART / "item_text_emb.npy"
IMAGE_EMB_P = ART / "item_image_emb.npy"
EMB_META_P = ART / "embedding_meta.json"

BATCH = 256
DIM   = 128
TEXT_DIM = int(os.environ.get("TEXT_EMB_DIM", "384"))
IMAGE_DIM = int(os.environ.get("IMAGE_EMB_DIM", "1408"))
HEAD_PRETRAIN_EPOCHS = 2
HEAD_PRETRAIN_BS = 1024
HEAD_LR = 2e-3
HEAD_WEIGHT_DECAY = 1e-4
FAISS_CANDIDATES = 300
TEST_FRACTION = 0.2
RECALL_K = 100
NDCG_K = int(os.environ.get("NDCG_K", "10"))
DEVICE = torch.device("cpu")

WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "false").lower() in {"1", "true", "yes"}
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "plush_item_tower")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")
WANDB_MODEL_ARTIFACT = os.environ.get("WANDB_MODEL_ARTIFACT", "item-encoder")


def init_wandb(**extra_config):
    """Create a Weights & Biases run if not disabled."""
    if WANDB_DISABLED:
        return None
    config = {
        "batch_size": BATCH,
        "embedding_dim": DIM,
        "device": str(DEVICE),
    }
    config.update(extra_config)
    settings = wandb.Settings(start_method="thread")
    kwargs = {"project": WANDB_PROJECT, "settings": settings}
    if WANDB_ENTITY:
        kwargs["entity"] = WANDB_ENTITY
    if WANDB_RUN_NAME:
        kwargs["name"] = WANDB_RUN_NAME
    return wandb.init(config=config, tags=["item-tower", "wandb"], **kwargs)


def log_model_artifact(run):
    """Upload the important artifacts to W&B for provenance."""
    if run is None:
        return
    artifact = wandb.Artifact(WANDB_MODEL_ARTIFACT, type="model")
    tracked_files = [
        (ART / "item_encoder.pt", "item_encoder.pt"),
        (ART / "faiss.index", "faiss.index"),
        (ART / "item_emb.npy", "item_emb.npy"),
        (ART / "items.parquet", "items.parquet"),
        (ART / "events.parquet", "events.parquet"),
        (ART / "reranker.joblib", "reranker.joblib"),
        (TEXT_EMB_P, "item_text_emb.npy"),
        (IMAGE_EMB_P, "item_image_emb.npy"),
        (EMB_META_P, "embedding_meta.json"),
        (ART / "mappings.json", "mappings.json"),
        (USER_P, "users.parquet"),
        (ART / "meta.json", "meta.json"),
    ]
    for path, name in tracked_files:
        if path.exists():
            artifact.add_file(str(path), name=name)
    run.log_artifact(artifact)


def log_dataset_artifact(run, train_path, test_path, metadata=None):
    """Attach the cached train/test splits as a dataset artifact in W&B."""
    if run is None:
        return
    artifact = wandb.Artifact(
        name="event-splits",
        type="dataset",
        metadata=metadata or {},
    )
    artifact.add_file(str(train_path), name=train_path.name)
    artifact.add_file(str(test_path), name=test_path.name)
    run.log_artifact(artifact)


def concat_text(row):
    pieces = [row["title"] or "", str(row["desc"] or "")]
    tags = row["tags"] if isinstance(row["tags"], list) else []
    pieces.append(" ".join(tags))
    if isinstance(row.get("introduction"), str):
        pieces.append("BrandIntro: " + row["introduction"].split(".")[0])
    return " ".join(pieces)


# def build_text_embeddings(items: pd.DataFrame, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#     model = SentenceTransformer(model_name, device="cpu")
#     texts = items.apply(concat_text, axis=1).tolist()
#     embs  = model.encode(texts, batch_size=256, show_progress_bar=True, device="cpu", convert_to_numpy=True, normalize_embeddings=False)
#     return embs, model.get_sentence_embedding_dimension()

def build_text_embeddings(
    items: pd.DataFrame,
    model_name: str = "cached",
    embedding_dim: int = TEXT_DIM,
    seed: int | None = 42,
):
    """
    Load precomputed text embeddings from `artifacts/item_text_emb.npy`.
    Falls back to random embeddings for local dev if the file is missing.

    Returns
    -------
    tuple[np.ndarray, int]
        Embedding matrix and its dimensionality.
    """
    if TEXT_EMB_P.exists():
        arr = np.load(TEXT_EMB_P, allow_pickle=False).astype(np.float32)
        return arr, arr.shape[1]

    print("warning: item_text_emb.npy not found; building random embeddings")
    rng = np.random.default_rng(seed)
    num_items = len(items)
    embs = rng.normal(size=(num_items, embedding_dim)).astype(np.float32)
    return embs, embedding_dim


def load_image_embeddings(items: pd.DataFrame, embedding_dim: int = IMAGE_DIM) -> Tuple[np.ndarray, int]:
    """
    Load cached image embeddings or return zeros if unavailable.
    """
    if IMAGE_EMB_P.exists():
        arr = np.load(IMAGE_EMB_P, allow_pickle=False).astype(np.float32)
        return arr, arr.shape[1]
    if embedding_dim <= 0:
        return np.zeros((len(items), 0), dtype=np.float32), 0
    return np.zeros((len(items), embedding_dim), dtype=np.float32), embedding_dim


def _align_embedding_rows(arr: np.ndarray, target_rows: int, kind: str) -> np.ndarray:
    """Ensure embedding matrices match the number of items by truncating or padding."""
    if arr.shape[0] == target_rows:
        return arr
    if arr.shape[0] > target_rows:
        print(
            f"[warn] {kind} embeddings have {arr.shape[0]} rows, expected {target_rows}. Truncating to match items.",
            flush=True,
        )
        return arr[:target_rows]
    pad_rows = target_rows - arr.shape[0]
    print(
        f"[warn] {kind} embeddings have {arr.shape[0]} rows, expected {target_rows}. "
        f"Padding {pad_rows} zero rows.",
        flush=True,
    )
    pad = np.zeros((pad_rows, arr.shape[1]), dtype=arr.dtype)
    return np.vstack([arr, pad])


def prepare_tensors(items: pd.DataFrame, maps: dict, text_emb: np.ndarray):
    """
    Convert item metadata and embeddings into PyTorch tensors for model input.

    This function transforms tabular item attributes (brand, color, and price band)
    and precomputed text embeddings into tensor representations suitable for model
    training or inference. It maps categorical fields to integer IDs, encodes price
    bands as one-hot vectors, and converts all arrays to PyTorch tensors.

    Parameters
    ----------
    items : pd.DataFrame
        DataFrame containing item metadata with the following columns:
            - `'brand'` : Categorical brand name or identifier (can contain NaNs).
            - `'color'` : Categorical color name or identifier (can contain NaNs).
            - `'price_band'` : Integer bin index for the item's price.
    maps : dict
        Dictionary containing precomputed mapping objects and configuration values:
            - `'brand2id'` : Mapping from brand names to integer IDs.
            - `'color2id'` : Mapping from color names to integer IDs.
            - `'price_bins'` : Total number of discrete price bins (int).
    text_emb : np.ndarray
        Array of precomputed text embeddings of shape `(num_items, embedding_dim)`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
            
            - text_emb : Float tensor of shape `(num_items, embedding_dim)`
              representing the item text embeddings.
            
            - brand_id : Long tensor of shape `(num_items,)`
              containing integer brand IDs.
            
            - color_id : Long tensor of shape `(num_items,)`
              containing integer color IDs.
            
            - price_oneh : Float tensor of shape `(num_items, num_price_bins)`
              containing one-hot encoded price band vectors.

    Notes
    -----
    - Missing brand or color values are mapped to `0`.
    - `price_band` values are clipped to the valid range `[0, P-1]`
      before one-hot encoding.
    - The function assumes that `maps["price_bins"]` defines the number
      of possible price categories.

    Example
    -------
    X_text, X_brand, X_color, X_price = prepare_tensors(items, maps, text_emb)
    X_text.shape, X_brand.shape, X_color.shape, X_price.shape
    (1000, 384), (1000,), (1000,), (1000, 10)
    """
    print("preparing brand tensor")
    brand_id = items["brand"].fillna("").map(maps["brand2id"]).fillna(0).astype(int).to_numpy()
    
    print("preparing color tensor")
    color_id = items["color"].fillna("").map(maps["color2id"]).fillna(0).astype(int).to_numpy()
    
    P = maps["price_bins"]
    print("preparing price tensor")
    price_oneh = np.eye(P, dtype=np.float32)[items["price_band"].astype(int).clip(0, P-1).to_numpy()]
    print("returning")
    return (
        torch.tensor(text_emb, dtype=torch.float32),
        torch.tensor(brand_id, dtype=torch.long),
        torch.tensor(color_id, dtype=torch.long),
        torch.tensor(price_oneh, dtype=torch.float32)
    )

class PairDataset(Dataset):
    """
    Yields (user_id, pos_item_id). Precompute user histories; sample positives.
    """
    def __init__(self, events: pd.DataFrame, items_index: pd.Index):
        self.events = events.sort_values("ts")
        self.by_user = dict(tuple(self.events.groupby("user_id")))
        self.items_index = items_index

        self.samples = []
        for u, df in self.by_user.items():
            for iid in df["item_id"].unique():
                self.samples.append((u, iid))
        np.random.shuffle(self.samples)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def user_vector_from_history(u: str, by_user: dict, item_vecs: torch.Tensor, item_id2pos: dict) -> torch.Tensor:
    """
    Compute a user embedding vector based on their historical item interactions.

    This function builds a user representation by aggregating the embeddings of items
    the user has interacted with. Each item's contribution is weighted by both:
    - **Event type weight** (e.g., clicks, saves, or purchases)
    - **Temporal decay** (recent interactions have more influence)

    The resulting user vector is L2-normalized to unit length, making it suitable for
    similarity computations (e.g., cosine similarity with item vectors).

    Parameters
    ----------
    u : str
        User identifier.
    by_user : dict
        A dictionary mapping user IDs to pandas DataFrames containing their interaction history.
        Each DataFrame must include the following columns:
            - `'item_id'` : Identifier for the interacted item.
            - `'event'`   : Type of interaction (e.g., "product_click", "save", "buy_click").
            - `'ts'`      : Timestamp (`datetime64[ns]`) of the interaction.
    item_vecs : torch.Tensor
        Tensor of item embedding vectors of shape `(num_items, embedding_dim)`.
    item_id2pos : dict
        Mapping from `item_id` to its corresponding row index (position) in `item_vecs`.

    Returns
    -------
    torch.Tensor
        A 1D tensor of shape `(embedding_dim,)` representing the user's aggregated embedding.
        If the user has no history (cold start), returns a zero vector.

    Notes
    -----
    - Event weights are assigned as:
        * `"product_click"` → 1.0  
        * `"save"` → 2.0  
        * `"buy_click"` → 3.0  
        * Any unknown event → 1.0  
    - Temporal decay is computed as `exp(-days_since_event / 30)`, giving a 30-day half-life.
    - The aggregation is a weighted sum of item embeddings followed by L2 normalization.

    Example
    -------
    user_vec = user_vector_from_history("user_123", by_user, item_vecs, item_id2pos)
    user_vec.shape
    torch.Size([384])
    """
    df = by_user.get(u)
    if df is None or df.empty:  # cold start
        return torch.zeros(item_vecs.shape[1], dtype=torch.float32)

    # weights: event * recency
    now = df["ts"].max()
    dd  = (now - df["ts"]).dt.total_seconds() / (3600 * 24)
    decay = np.exp(-dd / 30.0)
    evw = df["event"].map({"product_click":1.0,"save":2.0,"buy_click":3.0}).fillna(1.0).to_numpy()
    w = decay * evw

    vs = []
    for item_id, ww in zip(df["item_id"].tolist(), w.tolist()):
        pos = item_id2pos.get(item_id)
        if pos is not None:
            vs.append(item_vecs[pos].numpy() * ww)
    if not vs:
        return torch.zeros(item_vecs.shape[1], dtype=torch.float32)
    v = torch.tensor(np.sum(vs, axis=0), dtype=torch.float32)
    return F.normalize(v, p=2, dim=0)


def compute_dcg(ranked_items: List[str], relevance: Dict[str, float], k: int) -> float:
    """Compute DCG@k given a ranked list and relevance dictionary."""
    dcg = 0.0
    for rank, item_id in enumerate(ranked_items[:k], start=1):
        rel = relevance.get(item_id)
        if rel is None or rel <= 0:
            continue
        dcg += (2**rel - 1.0) / math.log2(rank + 1)
    return dcg


def evaluate_retrieval_metrics(
    ev_test: pd.DataFrame,
    faiss_index,
    item_vecs: np.ndarray,
    by_user: dict,
    item_id2pos: dict,
    recall_k: int = RECALL_K,
    ndcg_k: int = NDCG_K,
) -> Tuple[float, float]:
    """
    Evaluate Recall@K and nDCG@k for the retrieval layer.
    """
    if ndcg_k <= 0:
        ndcg_k = 1

    idx2item = {pos: iid for iid, pos in item_id2pos.items()}
    search_k = max(recall_k, ndcg_k)

    recall_hits = 0
    recall_users = 0
    ndcg_total = 0.0
    ndcg_users = 0

    item_tensor = torch.tensor(item_vecs)

    for u, df in ev_test.groupby("user_id"):
        relevance: Dict[str, float] = {}
        for _, row in df.iterrows():
            iid = row["item_id"]
            if iid not in item_id2pos:
                continue
            rel = float(row.get("weight", 1.0))
            if math.isnan(rel):
                rel = 0.0
            relevance[iid] = max(relevance.get(iid, 0.0), rel)

        if not relevance:
            continue

        uvec = user_vector_from_history(u, by_user, item_tensor, item_id2pos).numpy().astype("float32")
        if not np.any(uvec):
            continue

        _, indices = faiss_index.search(uvec.reshape(1, -1), search_k)
        ranked_items = [idx2item[i] for i in indices[0] if i >= 0]

        recall_users += 1
        if any(item in ranked_items[:recall_k] for item in relevance):
            recall_hits += 1

        dcg = compute_dcg(ranked_items, relevance, ndcg_k)
        ideal_dcg = 0.0
        for rank, rel in enumerate(sorted(relevance.values(), reverse=True)[:ndcg_k], start=1):
            if rel <= 0:
                continue
            ideal_dcg += (2**rel - 1.0) / math.log2(rank + 1)

        if ideal_dcg > 0:
            ndcg_total += dcg / ideal_dcg
            ndcg_users += 1

    recall = recall_hits / recall_users if recall_users else 0.0
    ndcg = ndcg_total / ndcg_users if ndcg_users else 0.0
    return recall, ndcg


def build_faiss_ip_index(item_vecs: np.ndarray, path_index, path_emb):
    """
    Build a FAISS inner-product index for normalized item embeddings, with
    safety checks and cleaning to prevent segmentation faults.

    This function ensures the input embeddings are valid (float32, 2D,
    contiguous, and finite), performs L2 normalization, and replaces any
    degenerate zero vectors with small random unit vectors before indexing.
    It then constructs a FAISS `IndexFlatIP` (inner-product) index for
    cosine-similarity retrieval and saves both the FAISS index and the
    normalized embeddings to disk.

    Parameters
    ----------
    item_vecs : np.ndarray
        Array of item embeddings of shape `(num_items, embedding_dim)`.
        Must be convertible to `float32`. Any NaN or infinite values will
        be replaced with zeros.
    path_index : str or pathlib.Path
        Output path for the serialized FAISS index file.
    path_emb : str or pathlib.Path
        Output path for saving the normalized embeddings as a `.npy` file.

    Returns
    -------
    faiss.IndexFlatIP
        The constructed FAISS inner-product index containing all item vectors.

    Raises
    ------
    ValueError
        If `item_vecs` is not 2-dimensional.

    Notes
    -----
    - All input vectors are converted to `float32` and normalized to unit length.
    - Any zero-norm vectors are replaced by random unit vectors to prevent NaNs.
    - The resulting index supports cosine-similarity search via inner product.
    - `faiss.IndexFlatIP` performs exact (non-approximate) nearest-neighbor search.

    Example
    -------
    index = build_faiss_ip_index(item_vecs, "artifacts/faiss.index", "artifacts/item_emb.npy")
    D, I = index.search(item_vecs[:1], 10)  # top-10 similar items
    print(D.shape, I.shape)
    (1, 10) (1, 10)
    """
    import faiss
    # Ensure correct dtype and contiguity
    V = np.asarray(item_vecs, dtype="float32")
    if V.ndim != 2:
        raise ValueError(f"item_vecs must be 2D, got shape {V.shape}")
    V = np.ascontiguousarray(V)

    # Clean invalid values
    np.nan_to_num(V, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize with safe handling for zero norms
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    zero_mask = norms.squeeze(1) == 0.0
    norms[zero_mask] = 1.0
    V /= norms

    # Replace degenerate rows with random unit vectors
    if np.any(zero_mask):
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(zero_mask.sum(), V.shape[1])).astype("float32")
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        V[zero_mask] = noise

    # Build and populate FAISS index
    d = V.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(V)

    # Persist index and vectors
    faiss.write_index(index, str(path_index))
    np.save(path_emb, V)

    return index

    
def main():
    """
    End-to-end training and evaluation pipeline for a lightweight
    retrieval + reranking recommender.

    This routine:
      1) Loads items and user events from Parquet files.
      2) Builds (placeholder) text embeddings and prepares model tensors.
      3) Initializes and briefly “head-pretrains” an item-tower encoder by
         self-distilling toward text embeddings (stabilizes early training).
      4) Produces final item vectors with the item tower.
      5) Builds a FAISS IndexFlatIP (cosine via normalized inner product) over items.
      6) Splits events temporally into train/test per user.
      7) Trains a simple logistic-regression reranker on pointwise features
         (similarity + price fit).
      8) Evaluates Recall@K on the FAISS retrieval layer.
      9) Logs metrics to Weights & Biases and saves artifacts.

    Side effects
    ------------
    - Reads:
        * `DATA_P` (items parquet)
        * `EV_P` (events parquet)
        * `MAP_P` (JSON with maps like brand2id, color2id, price_bins)
    
    - Writes artifacts under `ART`:
        * `faiss.index` (FAISS IndexFlatIP)
        * `item_emb.npy` (normalized item vectors)
        * `reranker.joblib` (optional; only if both classes present)
        * `item_encoder.pt` (PyTorch model state dict)
        * `mappings.json` (the `maps` object)
        * `meta.json` (dimension + timestamp)

    Logging
    -------
    - If a Weights & Biases run is active (`wandb_run`), records
      configuration, pretrain loss by epoch, and final Recall@100 on the test split, and
      logs model artifacts.

    Assumptions
    -----------
    - `build_text_embeddings`, `prepare_tensors`, `ItemTower`, `build_faiss_ip_index`,
      `evaluate_retrieval_metrics`, and `user_vector_from_history` are available.
    - `DIM`, `TEXT_DIM`, `DEVICE`, `DATA_P`, `EV_P`, `MAP_P`, `ART` and
      helper functions (`init_wandb`, `log_model_artifact`) are defined.
    - Events contain columns: user_id, item_id, event, ts, price, brand.
    - Items contain item_id, brand, color, price, etc. required by `prepare_tensors`.

    Notes
    -------
    Item tower training (head pretrain): the model isn’t predicting a label yet
    it’s learning to output an item embedding that matches the text embedding for that item (self-distillation). 
    So it “predicts” a vector.

    Retrieval (FAISS stage): given a user’s history, 
    we build a user embedding and compute similarity scores (inner product ≈ cosine) to all item embeddings. 
    This yields a Top-K candidate list—these are scores, not probabilities.

    Reranking (logistic regression): 
    for each user–candidate pair we compute features (currently sim and price_fit) and train a classifier 
    to predict the probability that the user will interact with that item (label 1 in the train window). 
    This stage outputs P(interaction | features).

    Evaluation (Recall@K): we check whether any of a user’s held-out positive items appear in the Top-K retrieved set. 
    So the reported metric is Recall@K of the retrieval layer, not accuracy of the reranker.

    Returns
    -------
    None
        All results are produced as side effects (artifacts/logs).
    """
    import faiss

    # Initialize Weights & Biases run (may return None if disabled).
    wandb_run = init_wandb()

    try:
        # ---- 1) Load data ----------------------------------------------------
        items = pd.read_parquet(DATA_P)
        print(items.head())

        events = pd.read_parquet(EV_P)
        print(events.head())

        if not USER_P.exists():
            raise FileNotFoundError(f"{USER_P} missing. Run data prep to materialize user features.")
        users = pd.read_parquet(USER_P)

        maps = json.loads(MAP_P.read_text())
        embedding_meta = json.loads(EMB_META_P.read_text()) if EMB_META_P.exists() else {}

        # ---- 2) Load cached text/image embeddings ---------------------------
        text_emb, text_feature_dim = build_text_embeddings(items, embedding_dim=TEXT_DIM)
        text_emb = _align_embedding_rows(text_emb, len(items), "text")
        image_emb, image_feature_dim = load_image_embeddings(items, embedding_dim=IMAGE_DIM)
        image_emb = _align_embedding_rows(image_emb, len(items), "image")
        if image_feature_dim > 0:
            tower_inputs = np.concatenate([text_emb, image_emb], axis=1)
        else:
            tower_inputs = text_emb
        tower_input_dim = tower_inputs.shape[1]

        training_params = {
            "optimizer": "AdamW",
            "learning_rate": HEAD_LR,
            "weight_decay": HEAD_WEIGHT_DECAY,
            "pretrain_epochs": HEAD_PRETRAIN_EPOCHS,
            "pretrain_batch_size": HEAD_PRETRAIN_BS,
            "tower_embedding_dim": DIM,
            "text_embedding_dim": int(text_feature_dim),
            "image_embedding_dim": int(image_feature_dim),
            "tower_input_dim": int(tower_input_dim),
            "retrieval_topk_candidates": FAISS_CANDIDATES,
            "recall_k": RECALL_K,
            "reranker": {
                "type": "LogisticRegression",
                "max_iter": 300,
                "class_weight": "balanced",
                "features": [
                    "tower_sim",
                    "price_fit",
                    "click_text_sim",
                    "purchase_text_sim",
                    "click_img_sim",
                    "purchase_img_sim",
                ],
            },
        }

        # Log run-time config/metadata to W&B (optional but helpful for lineage).
        if wandb_run:
            wandb_run.config.update(
                {
                    "text_embedding_meta": embedding_meta.get("text"),
                    "image_embedding_meta": embedding_meta.get("image"),
                    "text_dim": int(text_feature_dim),
                    "image_dim": int(image_feature_dim),
                    "num_items": int(len(items)),
                    "num_events": int(len(events)),
                    "training_params": training_params,
                },
                allow_val_change=True,
            )

        # Prepare tensors for the model (text, brand, color, price one-hot).
        text_t, brand_t, color_t, price_t = prepare_tensors(items, maps, tower_inputs)

        # ---- 3) Initialize item-tower model and pretrain head ----------------
        print("initializing model")
        model = ItemTower(
            tower_input_dim,
            len(maps["brand2id"]),
            len(maps["color2id"]),
            maps["price_bins"],
            d=DIM
        ).to(DEVICE)

        # Lightweight “self-distillation” against text embeddings:
        # pull item-tower outputs toward frozen text embeddings (stabilization).
        optim = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
        bs = HEAD_PRETRAIN_BS
        for epoch in range(HEAD_PRETRAIN_EPOCHS):
            perm = torch.randperm(text_t.size(0))       # Shuffle items
            for i in range(0, len(perm), bs):
                idx = perm[i:i+bs]                      # Mini-batch indices

                # Forward pass on the batch
                out = model(text_t[idx], brand_t[idx], color_t[idx], price_t[idx])

                # Target is normalized text embedding (pad/truncate to DIM as needed)
                tgt = (
                    F.normalize(text_t[idx][:, :DIM], p=2, dim=-1)
                    if text_t.size(1) >= DIM
                    else F.normalize(F.pad(text_t[idx], (0, DIM - text_t.size(1))), p=2, dim=-1)
                )

                # Cosine-like loss: 1 - mean(dot(out, tgt))
                loss = 1 - (out * tgt).sum(dim=1).mean()

                # Optimize
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            # Log final batch loss of the epoch to W&B (if active)
            if wandb_run:
                wandb_run.log({
                    "pretrain_head_loss": float(loss.item()),
                    "epoch": epoch,
                    "phase": "head_pretrain",
                }, step=epoch)

        # ---- 4) Produce item vectors with the trained item tower -------------
        print("producing item vectors")
        with torch.no_grad(): 
            item_vecs = (
                model(text_t, brand_t, color_t, price_t)  # (N, DIM) torch tensor
                .cpu()
                .numpy()
                .astype("float32")
            )

        # Diagnostics to catch malformed data prior to FAISS ingestion
        print("dtype:", item_vecs.dtype, "shape:", item_vecs.shape, "contiguous:", item_vecs.flags['C_CONTIGUOUS'])
        print("has NaN:", np.isnan(item_vecs).any(), "has Inf:", np.isinf(item_vecs).any())
        print("min/max:", np.min(item_vecs), np.max(item_vecs))
        print("zero-norm rows:", np.sum(np.linalg.norm(item_vecs, axis=1) == 0))
        assert item_vecs.shape[1] == int(item_vecs.shape[1])  # trivial guardrail

        # ---- 5) Build FAISS index (cosine via IP on L2-normalized vectors) ---
        print("building faiss")
        index = build_faiss_ip_index(                    # Handles normalization/cleaning internally
            item_vecs, ART / "faiss.index", ART / "item_emb.npy"
        )
        
        # Map item_id -> row position in items (needed to decode FAISS indices)
        item_id2pos = {iid: i for i, iid in enumerate(items["item_id"].tolist())}

        # ---- 6) Temporal train/test split per user ---------------------------
        ev = events.sort_values("ts")                    # Ensure chronological order
        by_user = dict(tuple(ev.groupby("user_id")))     # Full history per user

        def mask_fraction(df, frac): 
            n = max(1, int(len(df) * frac))              # Ensure ≥1 row in test
            return df.iloc[-n:]                          # Take most recent fraction

        # For each user, test = most-recent TEST_FRACTION of their events; train = remainder
        test = ev.groupby("user_id", group_keys=False).apply(lambda df: mask_fraction(df, TEST_FRACTION))
        train = ev.drop(test.index)

        train_split_path = ART / "events_train.parquet"
        test_split_path = ART / "events_test.parquet"
        train.to_parquet(train_split_path)
        test.to_parquet(test_split_path)

        if wandb_run:
            wandb_run.log({
                "data/train_rows": int(len(train)),
                "data/test_rows": int(len(test)),
                "data/test_fraction": TEST_FRACTION,
            })
            log_dataset_artifact(
                wandb_run,
                train_split_path,
                test_split_path,
                metadata={
                    "test_fraction": TEST_FRACTION,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "notes": "Per-user temporal split with latest events reserved for testing.",
                },
            )

        # ---- 7) Build pointwise reranker training data -----------------------
        def _ensure_vec(val, dim):
            if dim == 0:
                return np.zeros(0, dtype=np.float32)
            if isinstance(val, np.ndarray):
                arr = val.astype(np.float32)
            elif isinstance(val, list):
                arr = np.array(val, dtype=np.float32)
            else:
                return np.zeros(dim, dtype=np.float32)
            if arr.size == 0:
                return np.zeros(dim, dtype=np.float32)
            if arr.shape[0] != dim:
                out = np.zeros(dim, dtype=np.float32)
                take = min(dim, arr.shape[0])
                out[:take] = arr[:take]
                arr = out
            return arr

        def _history_lookup(col, dim):
            if col not in users.columns or dim == 0:
                return {}
            return {
                row["user_id"]: _ensure_vec(row[col], dim)
                for _, row in users[["user_id", col]].iterrows()
            }

        def _safe_cos(a, b):
            if a.size == 0 or b.size == 0:
                return 0.0
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        click_text_map = _history_lookup("click_history_text_emb", text_feature_dim)
        purchase_text_map = _history_lookup("purchase_history_text_emb", text_feature_dim)
        click_img_map = _history_lookup("click_history_img_emb", image_feature_dim)
        purchase_img_map = _history_lookup("purchase_history_img_emb", image_feature_dim)
        zero_text = np.zeros(text_feature_dim, dtype=np.float32) if text_feature_dim else np.zeros(0, dtype=np.float32)
        zero_img = np.zeros(image_feature_dim, dtype=np.float32) if image_feature_dim else np.zeros(0, dtype=np.float32)

        # For each user in train: retrieve Top-FAISS_CANDIDATES by FAISS, label 1 if item later appears for that user.
        rows = []
        for u, dfu in train.groupby("user_id"):
            # Build user vector from *full* history (by_user), then to float32 ndarray
            uvec = user_vector_from_history(u, by_user, torch.tensor(item_vecs), item_id2pos).numpy().astype("float32")
            if not uvec.any(): 
                continue  # Skip cold/no-signal users

            # Retrieve Top-300 candidate items for this user
            D, I = index.search(uvec.reshape(1, -1), FAISS_CANDIDATES)
            cands = [i for i in I[0] if i >= 0]

            # Positive set = items this user interacted with in the *training* window
            pos = set(dfu["item_id"].unique().tolist())

            # Construct simple features per candidate
            for j in cands:
                iid = items.iloc[j]["item_id"]          # Candidate item id
                label = 1 if iid in pos else 0          # Pointwise label (click/save/buy in train)

                # Features: user–item similarity and price-fit (Gaussian around user's median price)
                tower_sim = float(np.dot(uvec, item_vecs[j]))
                brand = items.iloc[j]["brand"]         
                color = items.iloc[j]["color"]
                price = items.iloc[j]["price"]
                u_med = dfu["price"].median()
                price_fit = math.exp(-((price - (0 if math.isnan(u_med) else u_med)) ** 2) / (2 * (300 ** 2)))

                t_click = click_text_map.get(u, zero_text)
                t_purchase = purchase_text_map.get(u, zero_text)
                v_text = text_emb[j] if text_feature_dim else zero_text
                click_text_sim = _safe_cos(t_click, v_text)
                purchase_text_sim = _safe_cos(t_purchase, v_text)

                i_click = click_img_map.get(u, zero_img)
                i_purchase = purchase_img_map.get(u, zero_img)
                v_img = image_emb[j] if image_feature_dim else zero_img
                click_img_sim = _safe_cos(i_click, v_img)
                purchase_img_sim = _safe_cos(i_purchase, v_img)

                rows.append([
                    label,
                    tower_sim,
                    price_fit,
                    click_text_sim,
                    purchase_text_sim,
                    click_img_sim,
                    purchase_img_sim,
                ])
        
        # Assemble feature matrix X and label vector y
        if rows:
            X = np.array([r[1:] for r in rows], dtype=np.float32)
            y = np.array([r[0] for r in rows], dtype=np.int32)
        else:
            X = np.zeros((0, len(training_params["reranker"]["features"])), dtype=np.float32)
            y = np.zeros((0,), dtype=np.int32)

        # Train a logistic-regression reranker if both classes are present
        if len(np.unique(y)) > 1:
            clf = LogisticRegression(max_iter=300, class_weight="balanced").fit(X, y)
            dump(clf, ART / "reranker.joblib")          # Persist reranker
        else:
            clf = None                                   # Not enough signal to train

        # ---- 8) Evaluate retrieval Recall@100 on the held-out test split ----
        idx = faiss.read_index(str(ART / "faiss.index")) # Fresh read to mirror prod usage
        recall_test, ndcg_test = evaluate_retrieval_metrics(
            test,
            idx,
            item_vecs,
            by_user,
            item_id2pos,
            recall_k=RECALL_K,
            ndcg_k=NDCG_K,
        )
        print(f"Test Recall@{RECALL_K}: {recall_test:.4f}")
        print(f"Test nDCG@{NDCG_K}: {ndcg_test:.4f}")

        # Log metric to W&B (explicit test split) and store in summary
        if wandb_run:
            metric_name = f"test/recall_at_{RECALL_K}"
            ndcg_metric_name = f"test/ndcg_at_{NDCG_K}"
            wandb_run.log({
                metric_name: float(recall_test),
                ndcg_metric_name: float(ndcg_test),
            })
            wandb_run.summary[metric_name] = float(recall_test)
            wandb_run.summary[ndcg_metric_name] = float(ndcg_test)

        # ---- 9) Persist artifacts --------------------------------------------
        torch.save(model.state_dict(), ART / "item_encoder.pt")           # Item tower weights
        (ART / "mappings.json").write_text(json.dumps(maps, indent=2))    # Save maps for inference
        meta_payload = {
            "item_tower_dim": DIM,
            "tower_input_dim": tower_input_dim,
            "text_embedding_dim": int(text_feature_dim),
            "image_embedding_dim": int(image_feature_dim),
            "recall_at_k": RECALL_K,
            "ndcg_at_k": NDCG_K,
            "trained_at": time.time(),
        }
        (ART / "meta.json").write_text(json.dumps(meta_payload, indent=2))
        log_model_artifact(wandb_run)                                     # Upload to W&B, if active
        print("Artifacts saved to", ART)
    
    finally:
        # Cleanly close the W&B run even on exceptions
        if wandb_run:
            wandb_run.finish()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
