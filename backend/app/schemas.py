from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The article body text to analyze.")
    title: Optional[str] = Field(None, description="The headline of the article.")
    url: Optional[str] = Field(None, description="The URL of the article, if available.")

class URLAnalysisRequest(BaseModel):
    url: str = Field(..., description="The URL of the news article to scrap and analyze.")

class SettingsUpdate(BaseModel):
    gemini_api_key: str = Field(..., min_length=10, description="Google Gemini API Key.")

class SettingsResponse(BaseModel):
    gemini_api_key_configured: bool

class ClaimAnalysedSchema(BaseModel):
    claim: str
    status: str  # "verified", "unverified", "false"
    explanation: str

class AnalysisResponse(BaseModel):
    id: int
    title: Optional[str]
    url: Optional[str]
    text_content: str
    ml_score: float
    llm_score: float
    trust_score: float
    verdict: str
    bias_rating: str
    clickbait_score: int
    key_findings: List[str]
    claims_analysed: List[ClaimAnalysedSchema]
    created_at: str

class StatsTimelineItem(BaseModel):
    date: str
    count: int
    avg_score: float

class StatsResponse(BaseModel):
    total_scans: int
    reliable_count: int
    misleading_count: int
    fake_count: int
    avg_trust_score: float
    bias_breakdown: dict
    timeline: List[StatsTimelineItem]
