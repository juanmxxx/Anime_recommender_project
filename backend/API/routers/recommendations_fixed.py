# Router for anime recommendations
from fastapi import APIRouter, Query, HTTPException, Depends, Body, Request
from sqlalchemy.orm import Session
import traceback
import time

from schemas.anime import (
    RecommendationRequest, MetricEvent
)
from services.anime import AnimeRecommendationService, MetricsService
from config.database import get_db

# Create router
router = APIRouter()

@router.get("/", summary="Get anime recommendations")
async def recommend(
    keywords: str = Query(..., description="Keywords or description to search animes"), 
    top_n: int = Query(5, description="Maximum number of recommendations"),
    session_id: str = Query(None, description="Session ID for metrics tracking"),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Endpoint to get anime recommendations based on keywords or descriptions.
    
    Args:
        keywords: Keywords or description to search animes
        top_n: Maximum number of recommendations to return
        session_id: Session ID for tracking metrics
        db: Database session
        request: FastAPI request object
        
    Returns:
        JSON with recommendations or error message
    """
    start_time = time.time()
    
    try:
        print(f"Processing recommendation request with keywords: {keywords}")
        
        # Create service and get recommendations
        service = AnimeRecommendationService(db)
        parsed_data = service.get_recommendations(keywords, top_n)
        
        # Record metrics if session_id is provided
        if session_id:
            try:
                end_time = time.time()
                load_time_ms = int((end_time - start_time) * 1000)
                
                metrics_service = MetricsService(db)
                metrics_service.record_metric(
                    MetricEvent(
                        session_id=session_id,
                        event_type="search",
                        prompt_text=keywords,
                        load_time_ms=load_time_ms
                    ),
                    request
                )
                print(f"✓ Search metrics recorded for session: {session_id}")
            except Exception as e:
                print(f"❌ Error recording metrics: {e}")
                traceback.print_exc()
        
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
async def get_recommendations(
    request: RecommendationRequest, 
    db: Session = Depends(get_db),
    fastapi_request: Request = None
):
    """POST endpoint to get recommendations based on keywords"""
    return await recommend(
        keywords=request.keywords, 
        top_n=request.top_n, 
        session_id=request.session_id, 
        db=db,
        request=fastapi_request
    )

@router.post("/metrics", summary="Record metrics")
def record_metrics(
    metric: MetricEvent,
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Endpoint to record metrics
    
    Args:
        metric: Metric event data
        db: Database session
        request: FastAPI request object
        
    Returns:
        Success status
    """
    try:
        metrics_service = MetricsService(db)
        metrics_service.record_metric(metric, request)
        return {"success": True, "message": "Metric recorded successfully"}
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error recording metrics: {error_msg}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": f"Error recording metrics: {error_msg}"}
        )
