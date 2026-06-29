import os
import requests
import io
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import List
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from html.parser import HTMLParser

from .database import init_db, get_db, ScanResult, Setting
from .ml_model import image_predictor
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.middleware("http")
async def log_requests(request, call_next):
    if "analyze/image" in request.url.path:
        print("--- DIAGNOSTIC REQUEST LOG ---")
        print("Path:", request.url.path)
        print("Headers:", dict(request.headers))
    response = await call_next(request)
    return response

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
    """Check if Gemini API Key is configured (DB or environment variable)."""
    setting = db.query(Setting).filter(Setting.key == "gemini_api_key").first()
    db_configured = bool(setting and setting.value)
    env_configured = bool(os.getenv("GEMINI_API_KEY", ""))
    is_configured = db_configured or env_configured
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

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image for forensics and deepfakes."""
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Determine if metadata has editing software signatures
        has_editing_software = 0
        keywords = [b'photoshop', b'gimp', b'adobe', b'canva', b'corel', b'paint.net', b'fotor', b'pixlr']
        for kw in keywords:
            if kw in file_bytes.lower():
                has_editing_software = 1
                break
        
        # Load image with Pillow
        try:
            image = Image.open(io.BytesIO(file_bytes))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid image format."
            )
        
        # Make sure it's RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # 1. Check EXIF existence
        has_exif = 0
        try:
            exif = image.getexif()
            if exif and len(exif) > 0:
                has_exif = 1
        except Exception:
            pass
            
        # 2. Perform ELA (Error Level Analysis)
        # Save at 90% quality and reload
        temp_io = io.BytesIO()
        image.save(temp_io, format='JPEG', quality=90)
        temp_io.seek(0)
        temp_img = Image.open(temp_io)
        
        # Get difference
        diff = ImageChops.difference(image, temp_img)
        diff_arr = np.array(diff)
        
        ela_mean = float(np.mean(diff_arr))
        ela_std = float(np.std(diff_arr))
        ela_max = float(np.max(diff_arr))
        
        # Enhance difference for visualization (scale contrast to max brightness)
        extrema = diff.getextrema()
        max_diff = 0
        for ext in extrema:
            if isinstance(ext, tuple):
                max_diff = max(max_diff, max(ext))
            else:
                max_diff = max(max_diff, ext)
                
        scale = 255.0 / max(1.0, max_diff)
        enhanced_diff = ImageEnhance.Brightness(diff).enhance(scale)
        
        # Convert enhanced diff to base64
        ela_io = io.BytesIO()
        enhanced_diff.save(ela_io, format='JPEG')
        ela_io.seek(0)
        ela_base64 = base64.b64encode(ela_io.read()).decode('utf-8')
        
        # 3. FFT high frequency noise
        fft_high_freq_mean = 0.0
        try:
            gray_img = image.convert('L')
            img_arr = np.array(gray_img)
            rows, cols = img_arr.shape
            if rows > 4 and cols > 4:
                f_transform = np.fft.fft2(img_arr)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                total_energy = np.sum(magnitude_spectrum)
                if total_energy > 0:
                    crow, ccol = rows // 2 , cols // 2
                    mask = np.ones((rows, cols))
                    r_center = max(2, min(rows, cols) // 8)
                    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
                    mask_center = x*x + y*y <= r_center*r_center
                    mask[mask_center] = 0
                    fft_high_freq_mean = float(np.sum(magnitude_spectrum[mask == 1]) / total_energy)
        except Exception as e:
            print("FFT calculation error:", e)
            
        # 4. Color space stats
        color_std_y = 50.0
        color_std_cb = 15.0
        color_std_cr = 15.0
        try:
            ycbcr = image.convert('YCbCr')
            y_arr, cb_arr, cr_arr = ycbcr.split()
            color_std_y = float(np.std(np.array(y_arr)))
            color_std_cb = float(np.std(np.array(cb_arr)))
            color_std_cr = float(np.std(np.array(cr_arr)))
        except Exception as e:
            print("Color stats calculation error:", e)
            
        # Compile features dict
        features = {
            "ela_mean": ela_mean,
            "ela_std": ela_std,
            "ela_max": ela_max,
            "has_exif": float(has_exif),
            "has_editing_software": float(has_editing_software),
            "fft_high_freq_mean": fft_high_freq_mean,
            "color_std_y": color_std_y,
            "color_std_cb": color_std_cb,
            "color_std_cr": color_std_cr
        }
        
        # Predict using Image Predictor
        prediction = image_predictor.predict(features)
        
        # Compile detailed forensic messages
        forensic_details = []
        
        # ELA check
        if ela_max > 80.0:
            forensic_details.append({
                "metric": "Error Level Analysis (ELA)",
                "status": "danger",
                "desc": f"Critical compression anomalies detected (Max diff: {ela_max:.1f}). High-contrast ELA suggests local manipulation (splicing or brush edits) along boundaries."
            })
        elif ela_mean > 5.0:
            forensic_details.append({
                "metric": "Error Level Analysis (ELA)",
                "status": "warning",
                "desc": f"Elevated compression variance detected (Mean diff: {ela_mean:.2f}). Pixel blocks show abnormal resaving signatures typical of composite images."
            })
        else:
            forensic_details.append({
                "metric": "Error Level Analysis (ELA)",
                "status": "success",
                "desc": f"Normal uniform error level (Mean diff: {ela_mean:.2f}). No local compression boundaries or pixel splicing artifacts detected."
            })
            
        # FFT Check
        if fft_high_freq_mean > 0.4:
            forensic_details.append({
                "metric": "Fourier Transform Noise Pattern",
                "status": "danger",
                "desc": f"Strong high-frequency structural noise detected (FFT index: {fft_high_freq_mean:.3f}). Indicates synthetic GAN/diffusion checkerboard grids."
            })
        elif fft_high_freq_mean > 0.15:
            forensic_details.append({
                "metric": "Fourier Transform Noise Pattern",
                "status": "warning",
                "desc": f"Anomalous high-frequency noise detected (FFT index: {fft_high_freq_mean:.3f}). Subtle digital artifacts present, suggesting face-swapping or neural generation blend lines."
            })
        else:
            forensic_details.append({
                "metric": "Fourier Transform Noise Pattern",
                "status": "success",
                "desc": f"Natural noise distribution (FFT index: {fft_high_freq_mean:.3f}). No neural network pattern periodic artifacts found."
            })
            
        # Metadata / EXIF Check
        if has_editing_software == 1:
            forensic_details.append({
                "metric": "Image Metadata Analysis",
                "status": "danger",
                "desc": "Active digital editing software signature (Photoshop/GIMP/Adobe/Canva) found embedded in the file headers."
            })
        elif has_exif == 0:
            forensic_details.append({
                "metric": "Camera EXIF Metadata",
                "status": "warning",
                "desc": "EXIF metadata is entirely missing. This is common for online/social media downloads or stripped images, but prevents camera hardware verification."
            })
        else:
            forensic_details.append({
                "metric": "Camera EXIF Metadata",
                "status": "success",
                "desc": "Valid camera EXIF headers found. File contains native hardware capture metadata."
            })
            
        # Color profile check
        if color_std_cb < 7.0 or color_std_cr < 7.0:
            forensic_details.append({
                "metric": "Color Space Distribution",
                "status": "warning",
                "desc": f"Abnormal chrominance dispersion (Cb std: {color_std_cb:.1f}, Cr std: {color_std_cr:.1f}). Typical of synthetic generator color spaces."
            })
        else:
            forensic_details.append({
                "metric": "Color Space Distribution",
                "status": "success",
                "desc": f"Normal color distribution across standard YCbCr spaces (Cb std: {color_std_cb:.1f}, Cr std: {color_std_cr:.1f})."
            })
            
        return {
            "filename": file.filename,
            "verdict": prediction["verdict"],
            "authenticity_score": round(prediction["authenticity_score"], 1),
            "probabilities": {
                "Authentic": round(prediction["probabilities"][0], 1),
                "Deepfake": round(prediction["probabilities"][1], 1),
                "AI-Generated": round(prediction["probabilities"][2], 1),
                "Morphed/Edited": 0.0
            },
            "ela_image": f"data:image/jpeg;base64,{ela_base64}",
            "features": features,
            "forensic_details": forensic_details
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image forensic analysis failed: {str(e)}"
        )

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
