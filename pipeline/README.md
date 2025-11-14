# ML Training Pipeline

## Training with Weights & Biases

1. Make sure `WANDB_API_KEY` is exported in your shell (the trainer will also pick up optional `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`, and `WANDB_MODEL_ARTIFACT`).
2. From `pipeline/`, run the trainer. Example:

   ```bash
   WANDB_PROJECT=plush-towers WANDB_RUN_NAME=local-debug uv run python -m src.train
   ```

During the run we log key metrics (`pretrain_head_loss`, `recall_at_50`) to both MLflow and W&B. After artifacts are written to `pipeline/artifacts/`, the same files are uploaded to a W&B model artifact (defaults to `item-encoder`). You can change the artifact name via `WANDB_MODEL_ARTIFACT`. The bundle now includes everything the FastAPI service needs (FAISS index, `item_emb.npy`, `items.parquet`, text/image `.npy` files, `users.parquet`, `reranker.joblib`, etc.).

Set `WANDB_DISABLED=true` if you need to skip online logging but still execute the training loop locally.

- Retrieval metrics now include both `Recall@K` and `nDCG@10` (configurable via `NDCG_K`). Both are logged to W&B summaries.

## Vertex AI Embeddings & Artifact Prep

1. Export the Vertex AI location + project you want to bill before running data prep:

   ```bash
   export EMBED_PROVIDER=vertex
   export VERTEX_PROJECT=my-gcp-project
   export VERTEX_LOCATION=us-central1
   export VERTEX_TEXT_MODEL=textembedding-gecko@001
   export VERTEX_IMAGE_MODEL=multimodalembedding@001
   ```

   If you omit these variables, the prep step falls back to deterministic random/zero embeddings so the rest of the pipeline still functions for local testing.

2. Generate the cached datasets and embedding artifacts (items/events/users parquet + `.npy` files for item text/image embeddings) once, then reuse them for training/runs:

   ```bash
   uv run python -m src.data_prep
   ```

3. The prep step writes embedding arrays to disk as soon as Vertex AI responses finish streaming, then reloads them to build pooled user-history embeddings (`click_history_*_emb`, `purchase_history_*_emb`). The API keeps these embeddings resident in-memory (no vector DB yet).
