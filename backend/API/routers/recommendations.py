# Router for anime recommendations
from fastapi import APIRouter, Query, HTTPException, Depends
from sqlalchemy.orm import Session
import traceback

from schemas.anime import RecommendationRequest
from services.anime import AnimeRecommendationService
from config.database import get_db

# Create router
router = APIRouter()

@router.get("/", summary="Get anime recommendations")
def recommend(
    keywords: str = Query(..., description="Keywords or description to search animes"), 
    top_n: int = Query(5, description="Maximum number of recommendations"),
    db: Session = Depends(get_db)
):
    """
    Endpoint to get anime recommendations based on keywords or descriptions.
    
    Args:
        keywords: Keywords or description to search animes
        top_n: Maximum number of recommendations to return
        db: Database session
        
    Returns:
        JSON with recommendations or error message
    """
    try:
        print(f"Processing recommendation request with keywords: {keywords}")
        
        # Create service and get recommendations
        service = AnimeRecommendationService(db)
        parsed_data = service.get_recommendations(keywords, top_n)
        
        print(f"Generated {len(parsed_data.get('recommendations', []))} recommendations")
        return parsed_data
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"API error: {error_msg}")
        print(stack_trace)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error in recommendation system: {error_msg}"
            }
        )

# POST endpoint for recommendations (maintain compatibility with previous API)
@router.post("/", summary="Get anime recommendations (POST)")
def get_recommendations(request: RecommendationRequest, db: Session = Depends(get_db)):
    """POST endpoint to get recommendations based on keywords"""
    return recommend(keywords=request.keywords, top_n=request.top_n, db=db)
