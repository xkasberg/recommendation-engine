from __future__ import annotations
"""
Recommender service that hydrates its entire retrieval + reranking stack from a
Weights & Biases model artifact.

Why two stages instead of a 26k-dimensional classifier?
-------------------------------------------------------
We have ~26k items. Training a single neural network with a 26k-way softmax
would require re-training whenever the catalog changes and would scale linearly
with the number of items per request. Instead we follow the “two-tower” pattern:

1. **Item tower (precomputed).** During training each item is encoded once into
   a dense vector. At inference those vectors live in memory and inside a FAISS
   index, so we can cosine-search the entire catalog in milliseconds.

2. **User tower approximation.** We build a user embedding on the fly by averaging the item vectors
   they interacted with, weighted by event-type and recency. This keeps inference
   stateless: if a user’s history changes, their vector changes automatically.

3. **FAISS candidate generation.** Given the user vector we retrieve the top
   `candidate_k` items (e.g., search 26k → keep 300). These candidates are now
   ordered by similarity but still fairly coarse.

4. **Pointwise reranker.** For every candidate we compute *features* (tower
   similarity, a price-fit score, cosine matches between the candidate’s text/
   image embeddings and the user’s pooled histories). Those features go into a
   small logistic regression that outputs a probability. Because the features
   are independent of absolute item IDs, the reranker works for any user/candidate
   pair—even if the catalog grows—and we avoid a 26k-wide output layer entirely.

Workflow overview:

1. Resolve a local artifact directory (either `ARTIFACTS_DIR` or a temp folder
   populated by downloading `WANDB_MODEL_ARTIFACT`).

2. Materialize the exact same assets produced by the training pipeline:
      - Normalized item embeddings + FAISS index for candidate generation.
      - Item metadata parquet (titles, images, prices, etc.).
      - Optional text/image embedding matrices for user-history similarity
        features.
      - Logistic-regression reranker (`reranker.joblib`), if present.

3. At request time, the service:
      - Converts raw event payloads to weighted interaction pairs
        (recency × event type).
      - Builds a user vector via weighted averaging of the item-tower vectors.
      - Uses FAISS to fetch top-N candidates and, if available, computes
        reranker features (tower sim, price fit, text/img cosine signals).
      - Returns the reranked top-k results with their metadata.

Because the artifacts are loaded once during startup, every request executes
purely in-memory: no DB fetches, no W&B round trips, and no additional model
instantiation. This mirrors exactly how the pipeline evaluated Recall/nDCG
so offline and online behavior stay aligned.
"""

import json
import logging
import math
import os
import pathlib
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
from joblib import load
import wandb

from models.recommender import Interaction

LOGGER = logging.getLogger(__name__)

REQUIRED_FILES = [
    "item_emb.npy",
    "faiss.index",
    "items.parquet",
    "mappings.json",
    "meta.json",
]


def _artifacts_present(path: pathlib.Path) -> bool:
    return all((path / fname).exists() for fname in REQUIRED_FILES)


def _artifact_spec() -> str:
    spec = os.environ.get("WANDB_MODEL_ARTIFACT_SPEC")
    if spec:
        return spec
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        raise ValueError("WANDB_PROJECT must be set to download artifacts from Weights & Biases.")
    entity = os.environ.get("WANDB_ENTITY")
    artifact_name = os.environ.get("WANDB_MODEL_ARTIFACT", "item-encoder")
    alias = os.environ.get("WANDB_MODEL_ARTIFACT_ALIAS", "latest")
    prefix = f"{entity}/{project}" if entity else project
    return f"{prefix}/{artifact_name}:{alias}"


class RecommenderService:
    """Memory-resident inference service backed by the W&B artifact bundle."""
    def __init__(self) -> None:
        """Download or reuse artifacts and hydrate all retrieval/ranking state."""
        self.artifacts_dir = self._resolve_artifacts()
        LOGGER.info("Loading recommender artifacts from %s", self.artifacts_dir)

        # --- Retrieval assets (shared for every request) ---------------------
        self.item_emb = np.load(self.artifacts_dir / "item_emb.npy", allow_pickle=False)
        self.index = faiss.read_index(str(self.artifacts_dir / "faiss.index"))
        self.items = pd.read_parquet(self.artifacts_dir / "items.parquet")
        self.item_id2pos = {iid: i for i, iid in enumerate(self.items["item_id"].tolist())}
        self.prices = self.items["price"].to_numpy()

        text_path = self.artifacts_dir / "item_text_emb.npy"
        image_path = self.artifacts_dir / "item_image_emb.npy"
        self.text_emb = np.load(text_path, allow_pickle=False) if text_path.exists() else None
        self.image_emb = np.load(image_path, allow_pickle=False) if image_path.exists() else None

        reranker_path = self.artifacts_dir / "reranker.joblib"
        self.reranker = load(reranker_path) if reranker_path.exists() else None

        meta = json.loads((self.artifacts_dir / "meta.json").read_text())
        self.dim = int(meta.get("item_tower_dim") or meta.get("dim") or self.item_emb.shape[1])

    def _resolve_artifacts(self) -> pathlib.Path:
        """Return a directory containing the serving bundle (local or W&B download)."""
        local_dir = os.environ.get("ARTIFACTS_DIR")
        if local_dir:
            path = pathlib.Path(local_dir).expanduser()
            if _artifacts_present(path):
                LOGGER.info("Using local artifact directory %s", path)
                return path
            LOGGER.warning("Local ARTIFACTS_DIR=%s missing required files. Falling back to W&B download.", path)

        spec = _artifact_spec()
        LOGGER.info("Downloading artifact %s from Weights & Biases", spec)
        api = wandb.Api()
        artifact = api.artifact(spec, type="model")
        target_root = pathlib.Path(tempfile.mkdtemp(prefix="plush-artifacts-"))
        artifact_path = pathlib.Path(artifact.download(root=str(target_root)))
        return artifact_path

    def user_vec_from_interactions(self, pairs: Sequence[Tuple[str, float]]) -> np.ndarray:
        """Average item vectors with event/recency weights to build a user embedding."""
        vs = []
        for iid, weight in pairs:
            pos = self.item_id2pos.get(iid)
            if pos is None:
                continue
            vs.append(self.item_emb[pos] * float(weight))
        if not vs:
            return np.zeros(self.dim, dtype="float32")
        stacked = np.sum(np.stack(vs, axis=0), axis=0)
        norm = np.linalg.norm(stacked) + 1e-12
        return (stacked / norm).astype("float32")

    def _prepare_pairs(self, interactions: Optional[List[Interaction]]) -> Tuple[List[Tuple[str, float]], List[str], List[str]]:
        """Convert raw interaction events into weighted pairs + click/purchase history."""
        if not interactions:
            return [], [], []

        now = datetime.now(timezone.utc)
        w_map = {"product_click": 1.0, "save": 2.0, "buy_click": 3.0}
        pairs: List[Tuple[str, float]] = []
        clicks: List[str] = []
        purchases: List[str] = []

        for inter in interactions:
            ts = None
            if inter.timestamp:
                try:
                    ts = datetime.fromisoformat(inter.timestamp.replace("Z", "+00:00"))
                except ValueError:
                    ts = None
            ts = ts or now
            days = max(0.0, (now - ts).total_seconds() / 86400.0)
            decay = math.exp(-days / 30.0)
            weight = w_map.get(inter.event, 1.0) * decay
            pairs.append((inter.item_id, weight))
            if inter.event == "product_click":
                clicks.append(inter.item_id)
            if inter.event == "buy_click":
                purchases.append(inter.item_id)

        return pairs, clicks, purchases

    def _aggregate_history(self, item_ids: List[str], matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Mean-pool history vectors for text/image similarity features."""
        if matrix is None or matrix.size == 0:
            return None
        vecs = [matrix[self.item_id2pos[iid]] for iid in item_ids if iid in self.item_id2pos]
        if not vecs:
            return None
        return np.mean(vecs, axis=0).astype("float32")

    @staticmethod
    def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _user_price_pref(self, item_ids: List[str]) -> float:
        """Median price of the user's history (falling back to global median)."""
        prices = [self.prices[self.item_id2pos[iid]] for iid in item_ids if iid in self.item_id2pos]
        if not prices:
            return float(np.nanmedian(self.prices))
        return float(np.median(prices))

    def recommend(
        self,
        interactions: Optional[List[Interaction]],
        *,
        k: int = 12,
        candidate_k: int = 300,
    ) -> List[Dict[str, object]]:
        """Run retrieval + reranking and return the top-k enriched item dicts."""
        pairs, clicks, purchases = self._prepare_pairs(interactions)
        uvec = self.user_vec_from_interactions(pairs)
        if not np.any(uvec):
            # Cold start: average item vector to keep FAISS happy
            uvec = np.mean(self.item_emb, axis=0).astype("float32")

        candidate_k = min(candidate_k, len(self.item_emb))
        sims = None
        D, I = self.index.search(uvec.reshape(1, -1), candidate_k)
        cand_idx = [idx for idx in I[0] if idx >= 0]
        if not cand_idx:
            return []

        # FAISS just narrowed ~26k items down to `candidate_k` highest-similarity vectors.
        # They remain sorted by cosine similarity, but we rerank them below using richer features.
        sims = self.item_emb[cand_idx] @ uvec

        click_text = self._aggregate_history(clicks, self.text_emb)
        purchase_text = self._aggregate_history(purchases, self.text_emb)
        click_img = self._aggregate_history(clicks, self.image_emb)
        purchase_img = self._aggregate_history(purchases, self.image_emb)
        user_price = self._user_price_pref([iid for iid, _ in pairs])

        rerank_scores = sims
        if self.reranker is not None and sims is not None:
            feature_rows = []
            for idx, sim in zip(cand_idx, sims):
                price = float(self.prices[idx])
                price_fit = math.exp(-((price - user_price) ** 2) / (2 * (300 ** 2)))

                text_vec = self.text_emb[idx] if self.text_emb is not None else None
                image_vec = self.image_emb[idx] if self.image_emb is not None else None

                # Feature vector mirrors the training pipeline: tower sim + price fit + text/image cosines
                feature_rows.append([
                    float(sim),
                    price_fit,
                    self._cosine(click_text, text_vec),
                    self._cosine(purchase_text, text_vec),
                    self._cosine(click_img, image_vec),
                    self._cosine(purchase_img, image_vec),
                ])

            X = np.asarray(feature_rows, dtype=np.float32)
            rerank_scores = self.reranker.predict_proba(X)[:, 1]

        order = np.argsort(-rerank_scores)[:k]
        recommendations = []
        for rank, local_idx in enumerate(order, start=1):
            item_row = self.items.iloc[cand_idx[local_idx]]
            recommendations.append(
                {
                    "item_id": item_row["item_id"],
                    "rank": rank,
                    "score": float(rerank_scores[local_idx]),
                    "sim": float(sims[local_idx]),
                    "title": item_row.get("title"),
                    "brand": item_row.get("brand"),
                    "price": float(item_row.get("price", 0.0)),
                    "color": item_row.get("color"),
                    "image": item_row.get("image"),
                    "explanations": ["Similarity to your history"],
                }
            )
        return recommendations
