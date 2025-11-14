
from typing import List, Optional

from pydantic import BaseModel, Field

class Interaction(BaseModel):
    item_id: str
    event: str
    timestamp: Optional[str] = None

class RecommendRequest(BaseModel):
    user_id: Optional[str] = None                 # optional; not used in this minimal bootstrap
    interactions: Optional[List[Interaction]] = None
    k: int = 12
    candidate_k: int = 300
    include_debug: bool = False
