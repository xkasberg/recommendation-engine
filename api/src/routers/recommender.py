from __future__ import annotations


from fastapi import APIRouter, HTTPException, Request

from models.recommender import RecommendRequest#, RecommendResponse

router = APIRouter(
    prefix="/v1/recommend",
    tags=["recommend"]
)

@router.post("")
def recommend(request: Request, req: RecommendRequest):
    """Proxy the request to the in-memory recommender service."""
    service = getattr(request.app.state, "recommender_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Recommender service unavailable")
    items = service.recommend(
        interactions=req.interactions,
        k=req.k,
        candidate_k=req.candidate_k,
    )
    return {"user": {"build": "inline" if req.interactions else "cache"},
            "items": items,
            "meta": {"k": req.k, "candidate_k": req.candidate_k}}
