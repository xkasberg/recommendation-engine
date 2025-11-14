# Recommendation API

## Configuration

The API downloads the latest trained model + embeddings from Weights & Biases at startup unless you point it to a local `ARTIFACTS_DIR`. Export the following before running:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=solway-ai           # optional if you own the project
export WANDB_PROJECT=plush-towers       # must match the trainer
export WANDB_MODEL_ARTIFACT=item-encoder
export WANDB_MODEL_ARTIFACT_ALIAS=latest
```

If you already have the artifact bundle on disk (e.g., from the pipeline run), set `ARTIFACTS_DIR=/path/to/pipeline/artifacts` to skip the download.

Install deps and start the service:

```bash
cd api
uv sync
uv run uvicorn src.main:app --reload --port 8080
```

## Example Request

```bash
curl -X POST http://localhost:8080/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{
        "user_id": "demo-user",
        "k": 12,
        "candidate_k": 300,
        "interactions": [
          {"item_id": "68ba0fdda97aed5b97de45af", "event": "product_click", "timestamp": "2025-10-01T23:36:59.894Z"},
          {"item_id": "68c34759a97aed5b97c4ce1d", "event": "buy_click", "timestamp": "2025-10-01T23:37:59.490Z"}
        ]
      }'
```

The response includes ranked items with similarity scores plus the metadata needed to render UI cards.

## How the Recommender Service Works

1. **Artifact hydration** – On startup the FastAPI lifespan hook instantiates `RecommenderService`, which either (a) reads the local `ARTIFACTS_DIR` or (b) downloads `WANDB_MODEL_ARTIFACT` (default `item-encoder:latest`) via the W&B API. The artifact contains the FAISS index, normalized item embeddings, item metadata parquet, optional text/image embedding matrices, and the logistic-regression reranker.

2. **Two-tower retrieval, not a 26k-way classifier** – Rather than emitting a 26k-dimensional softmax per request, we precompute one vector per item (the “item tower”). At inference the user vector is built on the fly by averaging the vectors of the products they touched, weighted by event type × exponential time decay. This keeps inference stateless and automatically adapts when new items arrive—only the offline pipeline needs to rebuild the FAISS index.

3. **Candidate narrowing** – The user vector queries the FAISS index and reduces 26k items down to `candidate_k` (e.g., 300) nearest neighbors. These candidates are already ordered by cosine similarity, which means we have a high-recall shortlist without running an enormous classifier each time.

4. **Pointwise reranker** – For every candidate we compute features that depend only on the user/item pair (tower similarity, a Gaussian price-fit score, cosine matches between the user’s pooled text/image histories and the candidate’s embeddings). Those features go through a logistic-regression model to produce a relevance score. Because the features are invariant to absolute item IDs, the reranker handles arbitrary catalogs and user subsets; we never need a fixed-size output layer.

5. **Response assembly** – The top `k` reranked items are returned with their metadata (title, brand, price, image) and diagnostics, so clients can render cards or debug recommendations. Everything lives in memory once startup completes, so per-request latency is bounded by FAISS + reranker math.
