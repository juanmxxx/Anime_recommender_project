from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class MetricEntry(Base):
    __tablename__ = 'user_metrics'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    event_type = Column(String, index=True, nullable=False)
    prompt_text = Column(Text, nullable=True)
    anime_clicked = Column(String, nullable=True)
    anime_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(INET, nullable=True)  # Usar INET para PostgreSQL
    load_time_ms = Column(Integer, nullable=True)
