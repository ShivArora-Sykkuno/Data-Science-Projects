from sqlalchemy import Column, Integer, String, Date, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DailyStats(Base):
    __tablename__ = "daily_stats"
    id = Column(Integer, primary_key=True)
    client_id = Column(String)
    platform = Column(String)
    date = Column(Date)
    followers = Column(Integer)
    impressions = Column(Integer)
    engagement_rate = Column(Float)
