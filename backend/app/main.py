import os
import requests
from typing import List
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from html.parser import HTMLParser

from .database import init_db, get_db, ScanResult, Setting
from .schemas import (
    TextAnalysisRequest,
    URLAnalysisRequest,
    SettingsUpdate,
    SettingsResponse,
    AnalysisResponse,
    StatsResponse,
    StatsTimelineItem
)
from .analyzer import analyze_article, analyze_blocked_url, get_api_key

app = FastAPI(title="Luffy AI Backend", version="1.0.0")

# Enable CORS for Chrome Extension calls and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows extension and external clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
def on_startup():
    init_db()

# HTML parser for clean text extraction from news websites
class NewsArticleExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.title_recording = False
        self.article_text = []
        self.title = ""
        self.ignore_tags = {"script", "style", "nav", "footer", "header", "aside", "form", "iframe"}
        self.active_tags = []

    def handle_starttag(self, tag, attrs):
        if tag in self.ignore_tags:
            self.active_tags.append(tag)
        elif tag == "title":
            self.title_recording = True
        elif tag in {"p", "h1", "h2", "h3", "article"}:
            if not any(t in self.ignore_tags for t in self.active_tags):
                self.recording = True

    def handle_endtag(self, tag):
        if tag in self.ignore_tags:
            if tag in self.active_tags:
                self.active_tags.remove(tag)
        elif tag == "title":
            self.title_recording = False
        elif tag in {"p", "h1", "h2", "h3", "article"}:
            self.recording = False

    def handle_data(self, data):
        if self.title_recording:
            self.title += data.strip()
        elif self.recording:
            text = data.strip()
            if len(text) > 15:  # Skip headers/junk
                self.article_text.append(text)

    def get_content(self):
        # Clean double spaces or multiple title records
        cleaned_title = re.sub(r'\s+', ' ', self.title).strip()
        return cleaned_title, "\n\n".join(self.article_text)

# Regex import for the extractor
import re

# ----------------- API Endpoints -----------------

@app.post("/api/analyze/text", response_model=AnalysisResponse)
def analyze_text(req: TextAnalysisRequest, db: Session = Depends(get_db)):
    """Analyze a pasted article manually."""
    try:
        result = analyze_article(db, req.title, req.text, req.url)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/api/analyze/url", response_model=AnalysisResponse)
def analyze_url(req: URLAnalysisRequest, db: Session = Depends(get_db)):
    """Fetch article content from URL and analyze."""
    api_key = get_api_key(db)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        res = requests.get(req.url, headers=headers, timeout=10)
        res.raise_for_status()
    except Exception as e:
        if api_key:
            try:
                result = analyze_blocked_url(db, req.url)
                return result.to_dict()
            except Exception as ai_err:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Scraping blocked. Live search check failed: {str(ai_err)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch content from URL (configure Gemini API Key to bypass social media blocks): {str(e)}"
            )

    try:
        extractor = NewsArticleExtractor()
        extractor.feed(res.text)
        title, text = extractor.get_content()

        if not text or len(text) < 100:
            # Fallback: if paragraph extraction yields nothing, use basic text cleanup
            # Strip html tags simple regex
            cleaned = re.sub('<[^<]+?>', '', res.text)
            # Remove scripts & styles
            cleaned = re.sub(r'(?i)<script.*?>.*?</script>', '', cleaned)
            cleaned = re.sub(r'(?i)<style.*?>.*?</style>', '', cleaned)
            text = "\n".join([line.strip() for line in cleaned.splitlines() if len(line.strip()) > 30])
            title = title or "Scraped Web Page"

        if not text or len(text) < 50:
            raise Exception("Article text content could not be cleanly extracted. Content is too short.")

        result = analyze_article(db, title, text, req.url)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process scraped web content: {str(e)}"
        )

@app.get("/api/history", response_model=List[AnalysisResponse])
def get_history(db: Session = Depends(get_db)):
    """Retrieve database scan logs."""
    scans = db.query(ScanResult).order_by(ScanResult.created_at.desc()).all()
    return [scan.to_dict() for scan in scans]

@app.delete("/api/history/{id}")
def delete_history_item(id: int, db: Session = Depends(get_db)):
    """Delete a scan record."""
    scan = db.query(ScanResult).filter(ScanResult.id == id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan log not found.")
    db.delete(scan)
    db.commit()
    return {"detail": "Scan history item deleted successfully."}

@app.get("/api/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Compile dashboard data & timeline metrics."""
    total = db.query(ScanResult).count()
    if total == 0:
        return StatsResponse(
            total_scans=0,
            reliable_count=0,
            misleading_count=0,
            fake_count=0,
            avg_trust_score=0.0,
            bias_breakdown={"Low": 0, "Moderate": 0, "High": 0, "Extreme": 0},
            timeline=[]
        )

    # Trust Score Averages
    avg_score = db.query(func.avg(ScanResult.trust_score)).scalar() or 0.0
    
    # Verdict metrics
    reliable = db.query(ScanResult).filter(ScanResult.trust_score >= 60).count()
    misleading = db.query(ScanResult).filter((ScanResult.trust_score >= 30) & (ScanResult.trust_score < 60)).count()
    fake = db.query(ScanResult).filter(ScanResult.trust_score < 30).count()

    # Bias Breakdown
    bias_map = {"Low": 0, "Moderate": 0, "High": 0, "Extreme": 0}
    bias_queries = db.query(ScanResult.bias_rating, func.count(ScanResult.id)).group_by(ScanResult.bias_rating).all()
    for rating, count in bias_queries:
        if rating in bias_map:
            bias_map[rating] = count

    # Timeline (last 7 days)
    timeline = []
    today = datetime.utcnow().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        
        # Query scans on this day (SQLite date function works nicely)
        day_scans = db.query(
            func.count(ScanResult.id),
            func.avg(ScanResult.trust_score)
        ).filter(
            func.date(ScanResult.created_at) == day_str
        ).first()

        count = day_scans[0] or 0
        avg_s = float(day_scans[1] or 0.0)
        
        timeline.append(StatsTimelineItem(
            date=day.strftime("%b %d"),
            count=count,
            avg_score=round(avg_s, 1)
        ))

    return StatsResponse(
        total_scans=total,
        reliable_count=reliable,
        misleading_count=misleading,
        fake_count=fake,
        avg_trust_score=round(avg_score, 1),
        bias_breakdown=bias_map,
        timeline=timeline
    )

@app.get("/api/settings", response_model=SettingsResponse)
def get_settings(db: Session = Depends(get_db)):
    """Check if Gemini API Key is configured."""
    setting = db.query(Setting).filter(Setting.key == "gemini_api_key").first()
    is_configured = bool(setting and setting.value)
    return SettingsResponse(gemini_api_key_configured=is_configured)

@app.post("/api/settings")
def update_settings(req: SettingsUpdate, db: Session = Depends(get_db)):
    """Save Gemini API Key."""
    setting = db.query(Setting).filter(Setting.key == "gemini_api_key").first()
    if setting:
        setting.value = req.gemini_api_key
    else:
        setting = Setting(key="gemini_api_key", value=req.gemini_api_key)
        db.add(setting)
    db.commit()
    return {"detail": "Gemini API Key updated successfully."}

# ----------------- Serves Web Dashboard -----------------

# Set directory path
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

# Mount assets directory if it exists
os.makedirs(FRONTEND_DIR, exist_ok=True)

@app.get("/{path:path}")
def serve_frontend(path: str):
    """Catch-all static server."""
    # Check if direct file request exists
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Fallback to serving main single page dashboard
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    # Return placeholder HTML if not created yet
    return FileResponse(index_path) if os.path.exists(index_path) else "Frontend files are currently building. Check back in a moment."
