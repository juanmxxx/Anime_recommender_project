from pydantic import BaseModel
from typing import Optional

# Schema for tracking metrics
class MetricEvent(BaseModel):
    session_id: str
    event_type: str  # 'search', 'click' or 'load_time'
    prompt_text: Optional[str] = None
    anime_clicked: Optional[str] = None
    anime_id: Optional[int] = None
    load_time_ms: Optional[int] = None  # Load time in milliseconds

# Schema for recommendation requests
class RecommendationRequest(BaseModel):
    keywords: str
    top_n: int = 5
