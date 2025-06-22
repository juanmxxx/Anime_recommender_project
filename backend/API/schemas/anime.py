from pydantic import BaseModel
from typing import Optional, List

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

# Schema classes for anime recommendations
class AnimeGenre(BaseModel):
    name: str

class AnimeBase(BaseModel):
    id: Optional[int] = None
    romaji_title: Optional[str] = None
    english_title: Optional[str] = None
    average_score: Optional[float] = None
    popularity: Optional[int] = None
    description: Optional[str] = None
    format: Optional[str] = None
    episodes: Optional[int] = None
    season_year: Optional[int] = None
    image_url: Optional[str] = None
    status: Optional[str] = None
    genres: List[str] = []

class AnimeRecommendation(AnimeBase):
    similarity: float = 0.0

class PromptRecommendationResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    query: Optional[str] = None
    recommendations: List[AnimeRecommendation] = []
    details: Optional[str] = None
