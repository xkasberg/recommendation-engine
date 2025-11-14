# Recommendation API curl Examples

All commands assume the API is running locally at `http://localhost:8080`. Adjust `BASE_URL` if you deploy elsewhere.

```bash
BASE_URL=${BASE_URL:-http://localhost:8080}
```

## 1. Basic Recommendation With Explicit Interactions

```bash
curl -X POST "$BASE_URL/v1/recommend" \
  -H 'Content-Type: application/json' \
  -d '{
        "user_id": "demo-user",
        "k": 12,
        "candidate_k": 300,
        "interactions": [
          {"item_id": "68ba0fdda97aed5b97de45af", "event": "product_click", "timestamp": "2025-10-01T23:36:59.894Z"},
          {"item_id": "68c34759a97aed5b97c4ce1d", "event": "buy_click", "timestamp": "2025-10-01T23:37:59.490Z"}
        ]
      }' | jq .
```

## 2. Cold-Start Request (No Interaction Payload)

```bash
curl -X POST "$BASE_URL/v1/recommend" \
  -H 'Content-Type: application/json' \
  -d '{
        "user_id": "new-user",
        "k": 8,
        "candidate_k": 200
      }' | jq .
```
