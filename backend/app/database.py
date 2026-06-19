import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create data directory relative to this file
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DB_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'fake_news.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ScanResult(Base):
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=True)
    url = Column(String, index=True, nullable=True)
    text_content = Column(Text, nullable=False)
    ml_score = Column(Float, nullable=False)        # Linguistic score (0-100% real)
    llm_score = Column(Float, nullable=False)       # Semantic score (0-100% real)
    trust_score = Column(Float, nullable=False)     # Combined score (0-100% real)
    verdict = Column(String, nullable=False)        # "Reliable", "Misleading", "Fake", etc.
    bias_rating = Column(String, nullable=False)    # "Low", "Moderate", "High", "Extreme"
    clickbait_score = Column(Integer, nullable=False)# 0-100
    key_findings = Column(Text, nullable=False)     # JSON serialized list of key findings
    claims_analysed = Column(Text, nullable=False)   # JSON serialized list of claims
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "text_content": self.text_content[:200] + "..." if len(self.text_content) > 200 else self.text_content,
            "ml_score": round(self.ml_score, 1),
            "llm_score": round(self.llm_score, 1),
            "trust_score": round(self.trust_score, 1),
            "verdict": self.verdict,
            "bias_rating": self.bias_rating,
            "clickbait_score": self.clickbait_score,
            "key_findings": json.loads(self.key_findings),
            "claims_analysed": json.loads(self.claims_analysed),
            "created_at": self.created_at.isoformat()
        }

class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True, index=True)
    value = Column(String, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
