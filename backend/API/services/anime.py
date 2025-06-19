import sys
import os
import json
import traceback
from sqlalchemy import func

# Add AI directory path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import AI modules
from AI.tools.APIprocessor import get_recommendations_from_prompt, CustomJSONEncoder
from models.anime import MetricEntry
from schemas.anime import MetricEvent, RecommendationRequest

class AnimeRecommendationService:
    def __init__(self, db=None):
        self.db = db
    
    def get_recommendations(self, keywords: str, top_n: int = 5):
        """
        Get anime recommendations based on keywords
        
        Args:
            keywords: Keywords or description to search animes
            top_n: Maximum number of recommendations to return
        
        Returns:
            Parsed recommendations data
        """
        try:
            # Call recommendation function
            results = get_recommendations_from_prompt(keywords, top_n)
            json_result = json.dumps(results, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            # Convert JSON string to Python object
            parsed_data = json.loads(json_result)
            
            return parsed_data
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"Error in recommendation service: {error_msg}")
            print(stack_trace)
            raise Exception(f"Error in recommendation system: {error_msg}")
            
class MetricsService:
    def __init__(self, db):
        self.db = db
        
    def record_metric(self, metric: MetricEvent, request=None):
        """
        Record a metric event in the database
        
        Args:
            metric: MetricEvent object with the data to record
            request: FastAPI Request object to extract user_agent and IP
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract request information if available
            user_agent = None
            ip_address = None
            if request:
                user_agent = request.headers.get("user-agent", None)
                ip_address = request.client.host if request.client else None
            
            # Create database entry from schema
            db_metric = MetricEntry(
                session_id=metric.session_id,
                event_type=metric.event_type,
                prompt_text=metric.prompt_text,
                anime_clicked=metric.anime_clicked,
                anime_id=metric.anime_id,
                load_time_ms=metric.load_time_ms,
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            # Add to database
            self.db.add(db_metric)
            self.db.commit()
            self.db.refresh(db_metric)
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Error recording metric: {e}")
            traceback.print_exc()  # Print full stack trace for debugging
            return False
    
    def get_metrics_summary(self):
        """
        Get summary of recorded metrics
        
        Returns:
            Dictionary with metrics summary
        """
        try:
            total_searches = self.db.query(MetricEntry).filter(MetricEntry.event_type == "search").count()
            total_clicks = self.db.query(MetricEntry).filter(MetricEntry.event_type == "click").count()
            avg_load_time = self.db.query(func.avg(MetricEntry.load_time_ms)).filter(
                MetricEntry.event_type == "load_time"
            ).scalar() or 0
            
            return {
                "total_searches": total_searches,
                "total_clicks": total_clicks,
                "average_load_time_ms": round(float(avg_load_time), 2),
            }
        except Exception as e:
            print(f"Error getting metrics summary: {e}")
            return {
                "error": str(e)
            }
