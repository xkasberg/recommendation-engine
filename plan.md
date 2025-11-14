# Plush “For You” — **CODEX AGENT PLAN**

A concrete, step-by-step plan (with runnable stubs) for an agent to implement a **two-tower retrieval + light rerank** recommendation service with:

- **`pipeline/`**: data prep → splits → CPU training → MLflow logging → artifacts
- **`api/`**: FastAPI inference service (Dockerized, CPU-only)
- **No frontend**. Inference is an HTTP endpoint.

---

## Repository Layout

```tree
.
├─ api/
│ ├─ app.py # FastAPI app + /v1/recommendations
│ ├─ service.py # load artifacts, user vec, recall, rerank, explain
│ ├─ mappings.json # (copied from artifacts/)
│ ├─ Dockerfile
│ └─ requirements.txt
├─ pipeline/
│ ├─ data/
│ │ ├─ brands.json
│ │ ├─ dresses.json
│ │ └─ events/
│ │ ├─ buy_click.csv
│ │ ├─ product_click.csv
│ │ └─ save.csv
│ ├─ data_prep.py # transforms → parquet (items/events/users)
│ ├─ towers.py # two-tower modules
│ ├─ train.py # train towers + ANN + reranker; log to MLflow
│ ├─ eval.py # recall@K / ndcg@K
│ ├─ requirements.txt
│ └─ run.sh # single entry for data→train→export
└─ artifacts/ # outputs (created by pipeline/train.py)
├─ item_encoder.pt
├─ user_aux.pt
├─ faiss.index
├─ item_emb.npy
├─ reranker.joblib
├─ items.parquet
├─ users.parquet
├─ mappings.json
└─ meta.json
```

---

1. **Parse & Normalize Data**
   - Read `pipeline/data/dresses.json` and `brands.json`.
   - Read event CSVs; select/rename columns → `(user_id, item_id, event, ts, price, brand)`.
   - Write `artifacts/items.parquet` and `artifacts/events.parquet`.
   - Build `users.parquet` (per-user aggregates) and `mappings.json` (id maps).

2. **Create Splits (temporal)**
   - For each user, sort by time; use earliest 70% → **train**, next 10% → **dev**, last 20% → **test**.
   - Persist split markers in `events.parquet` or `splits/*.parquet`.

3. **Train Two-Tower on CPU**
   - **Item tower**: frozen `sentence-transformers/all-MiniLM-L6-v2` text encoder + small MLP head to 128-d; add learned embeddings for brand/color; price band projection.
   - **User tower**: history-weighted mean of item embeddings (+ optional small MLP on facet histograms with residual add).
   - Loss: in-batch softmax (InfoNCE style). Train 1–2 epochs (CPU).
   - Log to **MLflow**: params, loss curves, Recall@50 / NDCG@10.

4. **Export Artifacts**
   - Embed all items → `item_emb.npy`; build FAISS (FlatIP or HNSW) → `faiss.index`.
   - Save towers (`item_encoder.pt`, `user_aux.pt`), `mappings.json`, and a **light reranker** (LogReg or XGB) → `reranker.joblib`.

5. **Serve Inference (FastAPI)**
   - Endpoint `POST /v1/recommendations` accepts either `{user_id}` or `{interactions:[...]}`.
   - Build user vector → ANN Top-K → rerank with interpretable features → return Top-k with explanations.

6. **Dockerize API**
   - CPU-only image; installs Python deps and exposes port 8080.

7. **Reproduce**
   - `pipeline/run.sh` runs end-to-end with MLflow tracking to `./mlruns` by default.

---


## Time Tracking

November 11th 2025: 3.25 Hours
10:30AM - 1:45PM

November 11th 2025: 2 Hours
9:30PM - 11:30PM