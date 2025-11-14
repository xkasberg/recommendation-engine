Training Pipeline & Recommendation API
======================================

This repository houses a full recommendation stack made of:

- `pipeline/` – data prep + training scripts that generate item/user embeddings, train the two-tower retrieval model + logistic reranker, evaluate Recall/nDCG, and upload a complete artifact bundle to Weights & Biases.
- `api/` – a FastAPI service that downloads the latest artifact bundle, warms all embeddings/indexes into memory, and serves personalized recommendations.

## Quick Start

1. **Data Prep**
   ```bash
   cd pipeline
   uv sync
   export EMBED_PROVIDER=vertex
   export VERTEX_PROJECT=your-gcp-project
   export VERTEX_LOCATION=us-central1
   uv run python -m src.data_prep
   ```

   This writes `artifacts/` containing parquet datasets, text/image embedding `.npy` files, user-history features, and metadata.

2. **Training**
   ```bash
   export WANDB_API_KEY=...
   uv run python -m src.train
   ```

   The trainer logs metrics (Recall@K, nDCG@10, loss curves) to W&B and pushes all serving artifacts in one model artifact (FAISS index, `item_emb.npy`, `items.parquet`, text/image embeddings, `users.parquet`, `reranker.joblib`, mapping files).

3. **API**
   ```bash
   cd ../api
   uv sync
   export WANDB_API_KEY=...
   export WANDB_PROJECT=plush-towers
   uv run uvicorn src.main:app --port 8080
   ```

   The service downloads the latest W&B artifact if `ARTIFACTS_DIR` isn’t set locally, then exposes `POST /v1/recommend`.

## Useful Docs

- `pipeline/README.md` – explains Vertex embedding config, training knobs, and artifact layout.
- `api/README.md` – covers the inference workflow, required env vars, and sample curl requests (`curls.md` contains ready-made CLI commands).

## Development Notes

- Embedding builders run asynchronously in batches with stdout-friendly progress lines.
- The recommender’s reranker consumes tower similarity, price-fit, and text/image cosine features so we can avoid a 26k-wide classifier layer.
- `.gitignore` excludes `artifacts/` and all generated assets; make sure to re-run data prep after changing the raw product/event files.
