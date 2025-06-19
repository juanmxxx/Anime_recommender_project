# Router for usage metrics
from fastapi import APIRouter, Request, Query, HTTPException, Depends
from sqlalchemy.orm import Session
import traceback

from schemas.anime import MetricEvent
from services.anime import MetricsService
from config.database import get_db

router = APIRouter()

@router.post("/search", summary="Record search event")
def log_search_event(request: Request, metric: MetricEvent, db: Session = Depends(get_db)):
    """Records search event"""
    service = MetricsService(db)
    success = service.record_metric(metric, request)
    return {"logged": success}

@router.post("/click", summary="Record anime click event")
def log_click_event(request: Request, metric: MetricEvent, db: Session = Depends(get_db)):
    """Records anime click event"""
    service = MetricsService(db)
    success = service.record_metric(metric, request)
    return {"logged": success}

@router.post("/load_time", summary="Record load time event")
def log_load_time_event(request: Request, metric: MetricEvent, db: Session = Depends(get_db)):
    """Records load time event"""
    service = MetricsService(db)
    success = service.record_metric(metric, request)
    return {"logged": success}

@router.get("/summary", summary="Get metrics summary")
def get_metrics_summary(db: Session = Depends(get_db)):
    """Gets metrics summary"""
    try:
        service = MetricsService(db)
        summary = service.get_metrics_summary()
        return summary
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error getting metrics: {error_msg}")
        print(stack_trace)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting metrics: {error_msg}"
        )
