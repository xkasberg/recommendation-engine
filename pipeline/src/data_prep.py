from __future__ import annotations
import asyncio
import io
import json
import logging
import math
import os
import pathlib
import re
import tempfile
import urllib.request
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import backoff
from google.api_core import exceptions as g_exceptions
import requests

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.preview.vision_models import Image as VertexImage
from vertexai.preview.vision_models import MultiModalEmbeddingModel

from src.utils.rate_limiter import RateLimiter

_VERTEX_IMAGE_RATE_LIMITER = RateLimiter(max_calls=595, period=60.0)

CLEAN_RE = re.compile(r"^.*?\.properties\.", re.I)

DATA_DIR = pathlib.Path(__file__).parent / "data" 
ART_DIR  = pathlib.Path(__file__).parents[2] / "artifacts"
ITEM_TEXT_EMB_PATH = ART_DIR / "item_text_emb.npy"
ITEM_IMAGE_EMB_PATH = ART_DIR / "item_image_emb.npy"
EMBED_META_PATH = ART_DIR / "embedding_meta.json"

ART_DIR.mkdir(exist_ok=True, parents=True)

EVENT_FILES = {
    "product_click": DATA_DIR / "events" / "product_click.csv",
    "save":          DATA_DIR / "events" / "save.csv",
    "buy_click":     DATA_DIR / "events" / "buy_click.csv",
}

LOGGER = logging.getLogger(__name__)

DEFAULT_TEXT_DIM = 768
DEFAULT_IMAGE_DIM = 1408
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "vertex").lower()
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "cs-poc-kkjv68ul5nbfhnxkuwkrvwd")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_TEXT_MODEL = os.environ.get("VERTEX_TEXT_MODEL", "gemini-embedding-001")

VERTEX_TEXT_TASK = os.environ.get("VERTEX_TEXT_TASK", "CLASSIFICATION")
VERTEX_TEXT_DIM = int(os.environ.get("VERTEX_TEXT_DIM", DEFAULT_TEXT_DIM))

VERTEX_TEXT_BATCH = int(os.environ.get("VERTEX_TEXT_BATCH", "250"))
VERTEX_TEXT_MAX_CONCURRENCY = int(os.environ.get("VERTEX_TEXT_MAX_CONCURRENCY", "4"))
VERTEX_IMAGE_MODEL = os.environ.get("VERTEX_IMAGE_MODEL", "multimodalembedding@001")
VERTEX_IMAGE_DIM = int(os.environ.get("VERTEX_IMAGE_DIM", DEFAULT_IMAGE_DIM))
IMAGE_BATCH_SIZE = 600
IMAGE_MAX_CONCURRENCY = int(os.environ.get("VERTEX_IMAGE_MAX_CONCURRENCY", "16"))

_VERTEX_READY = False

def _persist_embeddings(path: pathlib.Path, arr: np.ndarray) -> None:
    """Write embeddings array to disk immediately after it is generated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))
    LOGGER.info("Persisted embeddings to %s (shape=%s)", path, arr.shape)


def _concat_text_fields(row: pd.Series) -> str:
    pieces = [row.get("title") or "", row.get("desc") or ""]
    tags = row.get("tags")
    if isinstance(tags, list) and tags:
        pieces.append(" ".join(map(str, tags)))
    intro = row.get("introduction")
    if isinstance(intro, str) and intro:
        pieces.append("BrandIntro: " + intro.split(".")[0])
    brand = row.get("brand")
    if isinstance(brand, str) and brand:
        pieces.append(f"BrandName: {brand}")
    return " ".join(pieces).strip()


def _batch(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _ensure_vertex_initialized():
    global _VERTEX_READY
    if _VERTEX_READY:
        return
    if not VERTEX_PROJECT:
        raise EnvironmentError("VERTEX_PROJECT must be set to use Vertex AI embeddings.")
    try:
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        _VERTEX_READY = True
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Vertex AI: {exc}") from exc


def _random_embeddings(n_rows: int, dim: int, seed: int = 42) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    embs = rng.normal(size=(n_rows, dim)).astype(np.float32)
    meta = {
        "provider": "random",
        "dim": dim,
        "notes": "Fallback random embeddings (development only)",
    }
    return embs, dim, meta


@backoff.on_exception(
    backoff.expo,
    (
        g_exceptions.ResourceExhausted,   # quota / rate limit
        g_exceptions.TooManyRequests,     # 429
        g_exceptions.ServiceUnavailable,  # 503 / transient
    ),
    max_time=300,              # up to ~5 minutes per batch
    jitter=backoff.full_jitter,
)
def _vertex_get_text_embeddings_batch(
    model: TextEmbeddingModel,
    inputs: List[TextEmbeddingInput],
    **kwargs: Any,
):
    """
    Single Vertex *text* embedding batch call with exponential backoff
    on rate limits / transient errors.
    """
    return model.get_embeddings(inputs, **kwargs)


async def _vertex_text_embeddings_async(items: pd.DataFrame) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Asynchronously build text embeddings via Vertex AI with progress logging."""
    _ensure_vertex_initialized()

    model = TextEmbeddingModel.from_pretrained(VERTEX_TEXT_MODEL)
    LOGGER.info(
        "Building Vertex text embeddings via %s (task=%s, batch=%d, max_concurrency=%d)",
        VERTEX_TEXT_MODEL,
        VERTEX_TEXT_TASK,
        VERTEX_TEXT_BATCH,
        VERTEX_TEXT_MAX_CONCURRENCY,
    )

    texts = items.apply(_concat_text_fields, axis=1).tolist()
    total = len(texts)
    if total == 0:
        meta = {
            "provider": "vertex",
            "model": VERTEX_TEXT_MODEL,
            "task": VERTEX_TEXT_TASK,
            "dim": VERTEX_TEXT_DIM,
            "location": VERTEX_LOCATION,
        }
        return np.zeros((0, VERTEX_TEXT_DIM), dtype=np.float32), VERTEX_TEXT_DIM, meta

    kwargs: Dict[str, Any] = {}
    if VERTEX_TEXT_DIM:
        kwargs["output_dimensionality"] = VERTEX_TEXT_DIM

    num_batches = math.ceil(total / VERTEX_TEXT_BATCH)
    semaphore = asyncio.Semaphore(max(1, VERTEX_TEXT_MAX_CONCURRENCY))
    results: List[Optional[Tuple[int, List[np.ndarray]]]] = [None] * num_batches

    async def _embed_chunk(batch_idx: int, batch_texts: List[str]):
        async with semaphore:
            start_time = time.perf_counter()
            inputs = [TextEmbeddingInput(text, task_type=VERTEX_TEXT_TASK) for text in batch_texts]
            try:
                res = await asyncio.to_thread(_vertex_get_text_embeddings_batch, model, inputs, **kwargs)
            except Exception as exc:
                LOGGER.error("Vertex text embedding batch %d failed: %s", batch_idx + 1, exc)
                raise
            vectors = [np.array(r.values, dtype=np.float32) for r in res]
            results[batch_idx] = (batch_idx, vectors)
            duration = time.perf_counter() - start_time
            print(
                f"[text-emb] batch {batch_idx + 1}/{num_batches}: embedded {len(vectors)} items in {duration:.2f}s",
                flush=True,
            )

    tasks = []
    for batch_idx, start in enumerate(range(0, total, VERTEX_TEXT_BATCH)):
        batch_texts = texts[start : start + VERTEX_TEXT_BATCH]
        tasks.append(asyncio.create_task(_embed_chunk(batch_idx, batch_texts)))
    await asyncio.gather(*tasks)

    embeddings: List[np.ndarray] = []
    for entry in results:
        if entry is None:
            continue
        _, vectors = entry
        embeddings.extend(vectors)

    arr = np.vstack(embeddings).astype(np.float32)
    dim = arr.shape[1]
    LOGGER.info("Completed Vertex text embeddings (%d vectors, dim=%d)", arr.shape[0], dim)

    meta = {
        "provider": "vertex",
        "model": VERTEX_TEXT_MODEL,
        "task": VERTEX_TEXT_TASK,
        "dim": dim,
        "location": VERTEX_LOCATION,
    }
    return arr, dim, meta


def _vertex_text_embeddings(items: pd.DataFrame) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    return asyncio.run(_vertex_text_embeddings_async(items))


def fetch_image_bytes(url: str, timeout: float = 10.0) -> Optional[bytes]:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None
        
def _fetch_image_bytes(url: str, timeout: int = 10) -> bytes | None:
    """
    helper to read image from url
    """
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status >= 400:
                LOGGER.warning("Image fetch failed (status %s) for %s", resp.status, url)
                return None
            return resp.read()
    except Exception as exc:
        LOGGER.warning("Image fetch exception for %s: %s", url, exc)
        return None


@backoff.on_exception(
    backoff.expo,
    (
        g_exceptions.ResourceExhausted,   # often used for quota / rate-limit
        g_exceptions.TooManyRequests,     # explicit 429 where available
        g_exceptions.ServiceUnavailable,  # transient 503
    ),
    max_time=300,          # cap total wait ~5 minutes per call
    jitter=backoff.full_jitter,
)
def _vertex_get_image_embedding(
    model: MultiModalEmbeddingModel,
    vertex_image: VertexImage,
    dim: int,
):
    """Single Vertex image embedding call with exponential backoff on rate limits."""
    return model.get_embeddings(image=vertex_image, dimension=dim)


def _embed_single_image(
    model: MultiModalEmbeddingModel,
    dim: int,
    row: Dict[str, Any],
    global_idx: int,
) -> Optional[Tuple[int, np.ndarray]]:
    """Fetch bytes for a single item image and return its embedding vector."""
    raw = _fetch_image_bytes(row.get("image"))
    if not raw:
        LOGGER.warning("No image bytes for item_id=%s (idx=%d)", row.get("item_id"), global_idx)
        return None
    try:
        _VERTEX_IMAGE_RATE_LIMITER.acquire()
        with tempfile.NamedTemporaryFile(suffix=".img", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            vertex_image = VertexImage.load_from_file(tmp.name)
            result = _vertex_get_image_embedding(model, vertex_image, dim)
        if hasattr(result, "image_embedding"):
            vec = np.array(result.image_embedding, dtype=np.float32).flatten()
        else:
            vec = np.array(result.values, dtype=np.float32).flatten()
        out = np.zeros(dim, dtype=np.float32)
        take = min(dim, vec.size)
        out[:take] = vec[:take]
        return global_idx, out
    except Exception as exc:
        LOGGER.warning("Vertex image embedding failed for item_id=%s (idx=%d): %s", row.get("item_id"), global_idx, exc)
        return None


async def _vertex_image_embeddings_async(items: pd.DataFrame) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Asynchronously embed item images in batches with stdout-friendly progress."""
    _ensure_vertex_initialized()

    model = MultiModalEmbeddingModel.from_pretrained(VERTEX_IMAGE_MODEL)
    dim = VERTEX_IMAGE_DIM
    embs = np.zeros((len(items), dim), dtype=np.float32)
    total = len(items)
    LOGGER.info(
        "Building Vertex image embeddings via %s (dim=%d, total=%d, batch=%d)",
        VERTEX_IMAGE_MODEL,
        dim,
        total,
        IMAGE_BATCH_SIZE,
    )

    if total == 0:
        meta = {
            "provider": "vertex",
            "model": VERTEX_IMAGE_MODEL,
            "dim": dim,
            "location": VERTEX_LOCATION,
        }
        return embs, dim, meta

    semaphore = asyncio.Semaphore(max(1, IMAGE_MAX_CONCURRENCY))
    num_batches = math.ceil(total / IMAGE_BATCH_SIZE)

    async def _embed_row(global_idx: int, record: Dict[str, Any]):
        async with semaphore:
            return await asyncio.to_thread(_embed_single_image, model, dim, record, global_idx)

    for batch_idx, start in enumerate(range(0, total, IMAGE_BATCH_SIZE)):
        batch_records = items.iloc[start : start + IMAGE_BATCH_SIZE].to_dict("records")
        start_time = time.perf_counter()
        tasks = [
            asyncio.create_task(_embed_row(start + offset, record))
            for offset, record in enumerate(batch_records)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = 0
        for res in results:
            if isinstance(res, Exception) or res is None:
                if isinstance(res, Exception):
                    LOGGER.warning("Image embedding task exception: %s", res)
                continue
            idx, vec = res
            embs[idx] = vec
            success += 1
        duration = time.perf_counter() - start_time
        print(
            f"[image-emb] batch {batch_idx + 1}/{num_batches}: embedded {success}/{len(batch_records)} items in {duration:.2f}s",
            flush=True,
        )
        if success == 0:
            LOGGER.error(
                "Image embedding batch %d/%d produced zero vectors. Check Vertex credentials or rate limits.",
                batch_idx + 1,
                num_batches,
            )

    meta = {
        "provider": "vertex",
        "model": VERTEX_IMAGE_MODEL,
        "dim": dim,
        "location": VERTEX_LOCATION,
    }
    return embs, dim, meta


def _vertex_image_embeddings(items: pd.DataFrame) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Synchronous wrapper so callers can treat the async image builder as blocking."""
    return asyncio.run(_vertex_image_embeddings_async(items))


def build_text_embeddings(items: pd.DataFrame) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Build (or fall back to random) text embeddings for each item row.
    """
    if EMBED_PROVIDER == "vertex":
        try:
            arr, dim, meta = _vertex_text_embeddings(items)
            _persist_embeddings(ITEM_TEXT_EMB_PATH, arr)
            return arr, dim, meta
        except Exception as exc:
            LOGGER.warning("Vertex text embeddings unavailable (%s); falling back to random.", exc)
    LOGGER.warning("Falling back to random text embeddings (provider=%s disabled)", EMBED_PROVIDER)
    arr, dim, meta = _random_embeddings(len(items), DEFAULT_TEXT_DIM)
    _persist_embeddings(ITEM_TEXT_EMB_PATH, arr)
    return arr, dim, meta


def build_image_embeddings(items: pd.DataFrame) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Build main-image embeddings per item. Returns zeros if provider unavailable.
    """
    if EMBED_PROVIDER == "vertex":
        try:
            arr, dim, meta = _vertex_image_embeddings(items)
            _persist_embeddings(ITEM_IMAGE_EMB_PATH, arr)
            return arr, dim, meta
        except Exception as exc:
            LOGGER.warning("Vertex image embeddings unavailable (%s); using zeros.", exc)
    # deterministic zeros to keep downstream dimensionality stable
    dim = DEFAULT_IMAGE_DIM
    arr = np.zeros((len(items), dim), dtype=np.float32)
    LOGGER.warning("Using zero image embeddings (provider=%s disabled)", EMBED_PROVIDER)
    _persist_embeddings(ITEM_IMAGE_EMB_PATH, arr)
    return arr, dim, {
        "provider": "zeros",
        "dim": dim,
        "notes": "Image embeddings disabled or unavailable",
    }


def _load_embedding_matrix(path: pathlib.Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file missing: {path}. Run the embedding step first.")
    matrix = np.load(path, mmap_mode="r")
    if matrix.shape[0] != expected_rows:
        raise ValueError(f"Embedding shape mismatch for {path}: expected {expected_rows} rows, got {matrix.shape[0]}")
    return np.asarray(matrix, dtype=np.float32)


def pool_history_embeddings(
    users: pd.DataFrame,
    items: pd.DataFrame,
    text_emb_path: pathlib.Path = ITEM_TEXT_EMB_PATH,
    image_emb_path: pathlib.Path = ITEM_IMAGE_EMB_PATH,
) -> pd.DataFrame:
    """
    Attach pooled (mean) history embeddings for click/purchase histories.
    """
    id2idx = {iid: idx for idx, iid in enumerate(items["item_id"].tolist())}
    LOGGER.info("Loading text embeddings from %s", text_emb_path)
    text_emb = _load_embedding_matrix(text_emb_path, len(items))
    LOGGER.info("Loading image embeddings from %s", image_emb_path)
    image_emb = _load_embedding_matrix(image_emb_path, len(items))
    text_dim = text_emb.shape[1]
    image_dim = image_emb.shape[1]

    def _pool(ids: List[str], emb: np.ndarray, dim: int) -> List[float]:
        if dim == 0:
            return []
        vecs = [emb[id2idx[iid]] for iid in ids if iid in id2idx]
        if not vecs:
            return [0.0] * dim
        return np.mean(vecs, axis=0).astype(np.float32).tolist()

    users = users.copy()
    users["click_history_text_emb"] = users["click_history"].apply(lambda ids: _pool(ids, text_emb, text_dim))
    users["purchase_history_text_emb"] = users["purchase_history"].apply(lambda ids: _pool(ids, text_emb, text_dim))
    users["click_history_img_emb"] = users["click_history"].apply(lambda ids: _pool(ids, image_emb, image_dim))
    users["purchase_history_img_emb"] = users["purchase_history"].apply(lambda ids: _pool(ids, image_emb, image_dim))
    LOGGER.info("Attached pooled history embeddings (text_dim=%d, image_dim=%d)", text_dim, image_dim)
    return users


def write_embedding_artifacts(text_meta: Dict[str, Any], image_meta: Dict[str, Any]):
    """Persist embedding metadata (arrays are already written eagerly)."""
    meta = {
        "text": text_meta,
        "image": image_meta,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_items": text_meta.get("num_items") or image_meta.get("num_items") or None,
    }
    EMBED_META_PATH.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Wrote embedding metadata to %s", EMBED_META_PATH)


def _clean_cols(cols: List[str]) -> Dict[str,str]:
    out = {}
    for c in cols:
        cc = CLEAN_RE.sub("", c)  # drop "*.properties."
        cc = cc.replace("*.", "")  # drop "*."
        out[c] = cc
    return out


def _read_jsonl_or_array(p: pathlib.Path) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        if raw.startswith("["):
            return json.loads(raw)
        else:
            return [json.loads(line) for line in raw.splitlines()]


def load_items() -> pd.DataFrame:
    """
    Load and normalize product and brand data into a structured item DataFrame.

    This function reads product ("dresses.json") and brand ("brands.json") data
    from disk, flattens nested JSON structures, and extracts key fields such as
    item ID, brand, title, color, price, tags, description, and main image URL.
    It merges brand metadata (e.g., introductions) into the final item table,
    ensuring that each product has a consistent and deduplicated representation.

    Returns
    -------
    pd.DataFrame
        A normalized DataFrame containing item-level metadata with the following
        columns:

        - item_id : str  
          Unique identifier for each item.  
       
        - brand : str  
          Brand name of the item (may be empty if missing).  
        
        - title : str  
          Product name or title.  
        
        - desc : str  
          Concatenated textual description of the item from nested fields.  
       
        - tags : list[str]  
          List of categorical tags (empty list if missing).  
        
        - price : float  
          Numeric price value; non-numeric entries coerced to 0.  
       
       - color : str  
          Color label, if available.  
        
        - image : str or None  
          URL of the item’s primary image (prefers those marked `"main"`).  
        
        - introduction : str or NaN  
          Optional brand introduction text (joined via brand name).

    Notes
    -----
    - Input JSON files are expected under:  
      `DATA_DIR / "data/dresses.json"` and `DATA_DIR / "data/brands.json"`.
    - The helper `_read_jsonl_or_array` should return either a list of dicts
      or an array-like JSON object.
    - Price values that cannot be parsed as numeric are replaced with `0`.
    - Tags and descriptions are normalized to lists of strings.
    - Duplicate `item_id`s are dropped, and rows missing `item_id` are removed.
    - The function prints the shape of the raw items DataFrame for debugging.

    Example
    -------
    df_items = load_items()
    Num Items: (24972, 16)
    df_items.columns
    Index(['item_id', 'brand', 'title', 'desc', 'tags', 'price', 'color', 'image', 'introduction'], dtype='object')
    """

    dresses = _read_jsonl_or_array(DATA_DIR / "data/" / "dresses.json")
    brands  = _read_jsonl_or_array(DATA_DIR / "data/" / "brands.json")

    df_items = pd.json_normalize(dresses)
    print(f"Num Items: {df_items.shape}")
    
    # keys we care about
    df_items["item_id"]   = df_items["_id.$oid"]
    df_items["brand"]     = df_items.get("brand_name", "")
    df_items["title"]     = df_items.get("name", "")
    df_items["color"]     = df_items.get("color", "")
    df_items["price"]     = pd.to_numeric(df_items.get("price", 0), errors="coerce").fillna(0)
    df_items["tags"]      = df_items.get("tags", [[]]).apply(lambda x: x if isinstance(x, list) else [])
    
    # concat descriptions
    def _join_desc(xs):
        if not isinstance(xs, list): return ""
        return " ".join([str(x.get("text", "")) for x in xs if isinstance(x, dict)])
    
    df_items["desc"] = df_items.get("descriptions", []).apply(_join_desc).fillna("")
    
    # pick main image if present
    def _pick_img(imgs):
        if not isinstance(imgs, list) or not imgs: return None
        m = [im for im in imgs if im.get("title") == "main"]
        return (m[0]["url"] if m else imgs[0].get("url"))
    
    df_items["image"] = df_items.get("images", []).apply(_pick_img)

    df_items = df_items[["item_id", "brand", "title", "desc", "tags", "price", "color", "image"]].dropna(subset=["item_id"])
    df_items.drop_duplicates("item_id", inplace=True)

    # brand introductions (optional text boost)
    df_brands = pd.json_normalize(brands)[["name","introduction"]].rename(columns={"name":"brand"})
    df_items = df_items.merge(df_brands, on="brand", how="left")
    return df_items


def load_events() -> pd.DataFrame:
    """
    Load, normalize, and merge multiple user event logs into a unified DataFrame.

    This function reads a set of event log CSV files defined in `EVENT_FILES`,
    each representing a different event type (e.g., product clicks, saves,
    or purchases). It normalizes column names, deduplicates columns, and
    robustly extracts key fields such as user ID, item ID, timestamp, price,
    and brand, even when data inconsistencies exist (e.g., duplicated or
    missing columns).

    The resulting DataFrame provides a clean and chronologically ordered
    record of user–item interactions suitable for downstream modeling tasks
    such as recommendation training, recall evaluation, and user embedding
    construction.

    Returns
    -------
    pd.DataFrame
        A unified event-level DataFrame with the following columns:

        - user_id : str  
          Normalized user identifier (prefers `$user_id`, falls back to `distinct_id`).  
        
        - item_id : str  
          Product identifier from the event.  
        
        - event : str  
          Event name (e.g., `"product_click"`, `"save"`, `"buy_click"`).  
        
        - ts : datetime64[ns, UTC]  
          Event timestamp parsed from `timestamp` or `$time` fields.  
        
        - price : float  
          Numeric price associated with the event (NaNs coerced to `NaN`).  
        
        - brand : str or NA  
          Brand name recorded in the event.  
        
        - weight : float  
          Predefined event weight for downstream use  
          (`product_click` = 1.0, `save` = 2.0, `buy_click` = 3.0, default = 1.0).

    Notes
    -----
    - Event files are defined by the global dictionary `EVENT_FILES`, mapping
      event names to CSV file paths.
    - Column names are normalized via `_clean_cols`, and duplicates are removed.
    - The helper `_first_series` ensures robust extraction of a single column
      even when duplicates or missing values occur.
    - Timestamp fields are coerced to UTC datetimes and sorted ascending.
    - Missing or malformed rows lacking both `user_id` and `item_id` are dropped.
    - The computed `weight` column is used later in training and for constructing
      user embedding vectors.

    Example
    -------
    ev = load_events()
    events_df.shape
    (927, 7)
    ev.head(3)
                                user_id                   item_id          event                               ts  price       brand  weight
    73f0d97a-ea38-490e-8a4d-c76b5ca54552  68ba0fdda97aed5b97de45af  product_click 2025-10-01 23:36:59.894000+00:00    745        RIXO     1.0
    73f0d97a-ea38-490e-8a4d-c76b5ca54552  68c34759a97aed5b97c4ce1d  product_click 2025-10-01 23:37:17.362000+00:00   2225  Alex Perry     1.0
    73f0d97a-ea38-490e-8a4d-c76b5ca54552  68c34759a97aed5b97c4ce1d  product_click 2025-10-01 23:37:59.490000+00:00   2225  Alex Perry     1.0
    """

    frames = []

    def _first_series(df: pd.DataFrame, name: str) -> pd.Series:
        """Return a single Series for `name` even if duplicate columns exist or it's missing."""
        if name not in df.columns:
            return pd.Series(pd.NA, index=df.index)
        s = df[name]
        # If duplicates created a DataFrame, take the first occurrence
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s

    for ev_name, path in EVENT_FILES.items():
        df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
        df = df.rename(columns=_clean_cols(df.columns.tolist()))
        # 1) remove duplicate columns that can appear after renaming
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # 2) coalesce user id (prefer $user_id, fallback distinct_id)
        user_s = _first_series(df, "$user_id")
        distinct_s = _first_series(df, "distinct_id")
        user_id = user_s.combine_first(distinct_s)

        # 3) item id, timestamp and other fields (robust fallbacks)
        item_id = _first_series(df, "product_id")
        ts_raw = _first_series(df, "timestamp").combine_first(_first_series(df, "$time"))
        ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)

        price = pd.to_numeric(_first_series(df, "price"), errors="coerce")
        brand = _first_series(df, "brand")

        out = pd.DataFrame(
            {
                "user_id": user_id,
                "item_id": item_id,
                "event": ev_name,
                "ts": ts,
                "price": price,
                "brand": brand,
            }
        ).dropna(subset=["user_id", "item_id"])

        frames.append(out)

    ev = pd.concat(frames, ignore_index=True).sort_values("ts")
    # weights used later (training + user vec)
    ev["weight"] = ev["event"].map({"product_click": 1.0, "save": 2.0, "buy_click": 3.0}).fillna(1.0)
    return ev


def build_users(events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate user-level interaction statistics from an event log.

    This function summarizes user behavior across all recorded events by
    grouping the input `events` DataFrame by `user_id` and computing various
    interaction-level aggregates. The resulting DataFrame provides one row
    per user with counts of total events, event-type frequencies, the most
    recent activity timestamp, and median transaction price.

    Parameters
    ----------
    events : pd.DataFrame
        Event-level interaction DataFrame as returned by `load_events()`.
        Must include the following columns:
            - `'user_id'` : str — unique user identifier.
            - `'event'`   : str — type of event (e.g., `"product_click"`, `"save"`, `"buy_click"`).
            - `'ts'`      : datetime64[ns, UTC] — event timestamp.
            - `'price'`   : float — numeric price associated with the event.

    Returns
    -------
    pd.DataFrame
        A user-level summary DataFrame with one row per unique `user_id`
        and the following columns:

        - user_id : str  
          Unique identifier of the user.  
        
        - n_events : int  
          Total number of recorded events for the user.  
        
        - n_clicks : int  
          Count of `"product_click"` events.  
        
        - n_saves : int  
          Count of `"save"` events.  
        
        - n_buys : int  
          Count of `"buy_click"` events.  
        
        - last_ts : datetime64[ns, UTC]  
          Timestamp of the user's most recent recorded event.  
        
        - median_price : float  
          Median price of items interacted with by the user.
        
        - click_history : list[str]
          Chronologically ordered item_ids for product clicks (may be empty).

        - purchase_history : list[str]
          Chronologically ordered item_ids for buy events (may be empty).

    Notes
    -----
    - Event categories are hardcoded as `"product_click"`, `"save"`, and `"buy_click"`.
    - Users with no valid price values will have `median_price = NaN`.
    - The resulting table is commonly used to analyze user engagement patterns
      or to serve as a feature base for user-level embeddings.

    Example
    -------
    users = build_users(events)
    users.head(3)
                                      user_id  n_events  n_clicks  n_saves  n_buys                 last_ts  median_price
    0    01899331-38b2-4d3e-9641-c547d202ba0f        20        20        0       0 2025-10-16 22:53:07.012        1043.0
    1    03a66f7d-22ea-4f49-9f5d-f46c57c0f58c         4         4        0       0 2025-10-15 20:21:54.988        1150.5
    2    03aaa73c-f35f-4597-9e2c-fda21471cb9a         1         1        0       0 2025-10-05 23:27:07.394         952.0
    
    users_df.shape
    (170, 7)
    
    """
    ev = events.copy()

    # keep chronological order for histories
    ev.sort_values("ts", inplace=True)

    g = events.groupby("user_id")
    users = pd.DataFrame({
        "user_id": g.size().index,
        "n_events": g.size().values,
        "n_clicks": g.apply(lambda x: (x["event"]=="product_click").sum()).values,
        "n_saves":  g.apply(lambda x: (x["event"]=="save").sum()).values,
        "n_buys":   g.apply(lambda x: (x["event"]=="buy_click").sum()).values,
        "last_ts":  g["ts"].max().values,
        "median_price": g["price"].median().values
    })

    # Histories: collect item_ids per user by event type
    clicks = (
        ev[ev["event"] == "product_click"]
        .groupby("user_id")["item_id"]
        .agg(list)
    )
    
    buys = (
        ev[ev["event"] == "buy_click"]
        .groupby("user_id")["item_id"]
        .agg(list)
    )

    # Attach, defaulting to [] if a user has none
    users["click_history"] = users["user_id"].map(clicks).apply(lambda x: x if isinstance(x, list) else [])
    users["purchase_history"] = users["user_id"].map(buys).apply(lambda x: x if isinstance(x, list) else [])

    return users


def add_price_band(items: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    prices = items["price"].clip(lower=0)
    bins   = np.quantile(prices, np.linspace(0, 1, n_bins+1))
    bins[0] = 0.0  # robust floor
    items["price_band"] = np.digitize(prices, bins[1:-1], right=True)
    return items


def write_parquets(items: pd.DataFrame, events: pd.DataFrame, users: pd.DataFrame):
    """Save to parquet format"""
    items = add_price_band(items)
    items.to_parquet(ART_DIR / "items.parquet", index=False, engine="pyarrow")
    events.to_parquet(ART_DIR / "events.parquet", index=False, engine="pyarrow")
    users.to_parquet(ART_DIR / "users.parquet", index=False, engine="pyarrow")

    brands = {b:i for i,b in enumerate(sorted(items["brand"].fillna("").unique()))}
    colors = {c:i for i,c in enumerate(sorted(items["color"].fillna("").unique()))}
    price_bins = int(items["price_band"].max() + 1)
    mappings = {"brand2id": brands, "color2id": colors, "price_bins": price_bins}
    (ART_DIR / "mappings.json").write_text(json.dumps(mappings, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    LOGGER.info("Starting data prep pipeline")
    items  = load_items()
    LOGGER.info("Loaded %d items", len(items))
    events = load_events()
    LOGGER.info("Loaded %d events", len(events))
    users  = build_users(events)
    LOGGER.info("Built %d user rows", len(users))

    embed_items = items
    LOGGER.info("Getting %d Text Embeddings", len(embed_items))
    t0 = time.perf_counter()
    text_emb, _, text_meta = build_text_embeddings(embed_items)
    text_duration = time.perf_counter() - t0
    LOGGER.info("Text embedding build completed in %.2fs", text_duration)

    LOGGER.info("Getting %d Image Embeddings", len(embed_items))
    t1 = time.perf_counter()
    image_emb, _, image_meta = build_image_embeddings(embed_items)
    image_duration = time.perf_counter() - t1
    LOGGER.info("Image embedding build completed in %.2fs", image_duration)

    text_meta["num_items"] = len(embed_items)
    image_meta["num_items"] = len(embed_items)
    text_meta["duration_seconds"] = round(text_duration, 3)
    image_meta["duration_seconds"] = round(image_duration, 3)

    users_with_history = pool_history_embeddings(users, items)
    write_parquets(items, events, users_with_history)
    write_embedding_artifacts(text_meta, image_meta)
    LOGGER.info("Wrote artifacts and embeddings to %s", ART_DIR)
