import json
import os
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from .database import Setting, ScanResult
from .ml_model import predictor

# Define structured Pydantic schemas for Gemini API responses
class ClaimCheck(BaseModel):
    claim: str = Field(description="A significant factual claim extracted from the article.")
    status: str = Field(description="The status of the claim. Must be one of: 'verified', 'unverified', or 'false'.")
    explanation: str = Field(description="Brief explanation of why this status was assigned, citing general facts or common knowledge.")

class GeminiAnalysisResult(BaseModel):
    trustScore: int = Field(description="Semantic/factual trust rating from 0 (completely fabricated) to 100 (fully verified/accurate).")
    verdict: str = Field(description="Overall verdict. Must be one of: 'Reliable', 'Mostly Reliable', 'Plausible', 'Misleading', 'Fake'.")
    explanation: str = Field(description="A concise (2-3 sentences) overall summary explanation of the news reliability.")
    biasRating: str = Field(description="Political/rhetorical bias rating. Must be one of: 'Low', 'Moderate', 'High', 'Extreme'.")
    clickbaitScore: int = Field(description="Rating from 0 to 100 showing how sensational or clickbaity the headline/opening is.")
    keyFindings: List[str] = Field(description="List of 3-4 key logical or rhetorical findings (e.g., loaded language, source verification status, circular reasoning).")
    claimsAnalysed: List[ClaimCheck] = Field(description="List of 3-5 specific claims extracted and verified.")

def get_api_key(db: Session) -> str:
    """Retrieves the Gemini API Key from the database settings."""
    setting = db.query(Setting).filter(Setting.key == "gemini_api_key").first()
    if setting and setting.value:
        return setting.value
    # Fallback to environment variable
    return os.getenv("GEMINI_API_KEY", "")

def analyze_article(db: Session, title: str, text: str, url: str = None) -> ScanResult:
    """
    Main entry point for analyzing a news article.
    Combines:
      1. Syntactic/Linguistic analysis (ML Model)
      2. Semantic/Fact-checking analysis (Gemini API LLM)
      3. SQLite storage & final score calculation
    """
    # 1. Linguistic Stylistic Check (ML model)
    ml_score = predictor.predict(title, text)
    
    # 2. Get API Key
    api_key = get_api_key(db)
    
    llm_analysis = None
    if api_key:
        try:
            print("Invoking Gemini API for semantic fact-check...")
            client = genai.Client(api_key=api_key)
            
            prompt = f"""
            Analyze the following news article for credibility, political bias, clickbait levels, and fact-check its key claims.
            
            Headline: {title or 'N/A'}
            URL/Source: {url or 'N/A'}
            
            Article Text:
            {text}
            """
            
            # Request structured output matching GeminiAnalysisResult
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiAnalysisResult,
                        system_instruction=(
                            "You are an expert fact-checker and media credibility analyst. Analyze the article objectively. "
                            "Identify logical fallacies, verify claims against general factual knowledge up to your training cutoff, "
                            "evaluate political and emotional bias, and determine if the headline is clickbait."
                        )
                    ),
                )
            except Exception as model_err:
                print(f"gemini-2.5-flash model failed ({model_err}). Retrying with gemini-2.0-flash...")
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiAnalysisResult,
                        system_instruction=(
                            "You are an expert fact-checker and media credibility analyst. Analyze the article objectively. "
                            "Identify logical fallacies, verify claims against general factual knowledge up to your training cutoff, "
                            "evaluate political and emotional bias, and determine if the headline is clickbait."
                        )
                    ),
                )
            
            # The response text will be a valid JSON matching GeminiAnalysisResult schema
            llm_data = json.loads(response.text)
            llm_analysis = llm_data
            print("Gemini API call completed successfully.")
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}. Falling back to simulation.")
            llm_analysis = None

    # 3. Handle Fallback (if no API Key or API call failed)
    if not llm_analysis:
        llm_analysis = generate_simulated_analysis(title, text, ml_score)
        # Indicate in findings that it is simulated/demo mode
        llm_analysis["keyFindings"].insert(0, "Demo Mode: Gemini API key is not configured. This is a simulated fact-check.")

    # 4. Calculate Hybrid Combined Score
    # We weight: 30% Linguistic Style (ML), 70% Factual/Semantic Analysis (LLM)
    llm_score = float(llm_analysis["trustScore"])
    trust_score = (ml_score * 0.3) + (llm_score * 0.7)
    
    # Adjust verdict based on final trust score
    if trust_score >= 80:
        verdict = "Reliable"
    elif trust_score >= 60:
        verdict = "Mostly Reliable"
    elif trust_score >= 45:
        verdict = "Plausible but Biased"
    elif trust_score >= 25:
        verdict = "Misleading / Disinformation"
    else:
        verdict = "Fake News / Fabricated"

    # 5. Save to Database
    db_result = ScanResult(
        title=title or "Untitled Scan",
        url=url,
        text_content=text,
        ml_score=ml_score,
        llm_score=llm_score,
        trust_score=trust_score,
        verdict=verdict,
        bias_rating=llm_analysis["biasRating"],
        clickbait_score=int(llm_analysis["clickbaitScore"]),
        key_findings=json.dumps(llm_analysis["keyFindings"]),
        claims_analysed=json.dumps(llm_analysis["claimsAnalysed"])
    )
    
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    
    return db_result

def generate_simulated_analysis(title: str, text: str, ml_score: float) -> Dict[str, Any]:
    """Generates a realistic analysis report for bootstrapping / offline demo testing."""
    text_lower = text.lower()
    title_lower = (title or "").lower()
    
    # Basic heuristics to decide simulated values
    is_suspicious = ml_score < 50 or "!" in title_lower or any(w in text_lower for w in ["shocking", "aliens", "secret cure", "banned"])
    
    if is_suspicious:
        trust_score = int(ml_score * 0.8) # Keep it low if ML says so
        verdict = "Misleading"
        bias_rating = "High"
        clickbait_score = 75
        key_findings = [
            "Headline contains sensationalized or exaggerated language (clickbait style).",
            "Relies heavily on emotional appeals rather than verified primary sources.",
            "Text matches stylistic markers frequently seen in misleading or low-credibility reports."
        ]
        claims = [
            {
                "claim": f"Sensational claims made in: '{title[:50]}...'" if title else "Main article claim",
                "status": "unverified",
                "explanation": "No external primary sources, government announcements, or peer-reviewed citations were found to verify this."
            },
            {
                "claim": "The event described happened recently with universal consensus.",
                "status": "false",
                "explanation": "Stylistic structure is highly inconsistent with mainstream journalism practices."
            }
        ]
    else:
        trust_score = int(max(65.0, ml_score))
        verdict = "Mostly Reliable"
        bias_rating = "Moderate"
        clickbait_score = 20
        key_findings = [
            "Linguistic style is formal, informative, and structurally objective.",
            "Minimal use of exclamation marks or sensationalist vocabulary.",
            "Presents details in a standard journalistic format."
        ]
        claims = [
            {
                "claim": "General assertions in the news article",
                "status": "verified",
                "explanation": "Linguistic structure aligns with high-credibility news reports."
            }
        ]
        
    return {
        "trustScore": trust_score,
        "verdict": verdict,
        "explanation": "This report was generated using stylistic ML heuristics (Demo Mode). Set a Gemini API key in settings for real-time AI fact-checking.",
        "biasRating": bias_rating,
        "clickbaitScore": clickbait_score,
        "keyFindings": key_findings,
        "claimsAnalysed": claims
    }

def analyze_blocked_url(db: Session, url: str) -> ScanResult:
    """
    Handles URLs blocked by scraping limiters (like social media).
    Uses Gemini with Google Search Grounding to verify the URL contents and claims.
    """
    api_key = get_api_key(db)
    if not api_key:
        raise Exception("Gemini API Key is required to perform live search fact-checking.")

    try:
        print(f"Bypassing social media block. Performing Gemini Search Grounding on URL: {url}...")
        client = genai.Client(api_key=api_key)
        
        search_prompt = f"""
        Search the web to find the details, headline, publisher, and specific claims of this URL: {url}
        Perform a comprehensive fact-check on the claims made in this post/article.
        
        Provide:
        - What the post/article says.
        - The specific claims being made.
        - Whether each claim is verified, unverified, or false based on reliable news sources or facts.
        - The general rhetorical bias.
        - Clickbait/sensational level.
        """
        
        # 1. Fetch live web details
        try:
            search_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=search_prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                ),
            )
        except Exception as search_err:
            print(f"gemini-2.5-flash search failed: {search_err}. Retrying with gemini-2.0-flash...")
            search_response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=search_prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                ),
            )
        
        report_text = search_response.text
        print("Search details fetched. Translating to JSON schema...")
        
        # 2. Format to JSON
        format_prompt = f"""
        Format the following fact-checking report about a social media URL into the required structured JSON schema.
        
        Report:
        {report_text}
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=format_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiAnalysisResult,
                ),
            )
        except Exception as format_err:
            print(f"gemini-2.5-flash format failed: {format_err}. Retrying with gemini-2.0-flash...")
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=format_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiAnalysisResult,
                ),
            )
        
        llm_analysis = json.loads(response.text)
        print("Blocked URL analysis completed successfully.")
        
    except Exception as e:
        print(f"Failed to scan blocked URL via Gemini search grounding: {e}")
        raise Exception(f"AI Search check failed: {str(e)}")

    # Calculate hybrid score (ML stylistic check defaults to neutral 70 since text is inaccessible)
    ml_score = 70.0
    llm_score = float(llm_analysis["trustScore"])
    trust_score = (ml_score * 0.15) + (llm_score * 0.85) # Weight LLM heavily because ML did not see text

    if trust_score >= 80:
        verdict = "Reliable"
    elif trust_score >= 60:
        verdict = "Mostly Reliable"
    elif trust_score >= 45:
        verdict = "Plausible but Biased"
    elif trust_score >= 25:
        verdict = "Misleading / Disinformation"
    else:
        verdict = "Fake News / Fabricated"

    # Extract dynamic title from search summary
    title = llm_analysis.get("explanation", "")[:80] + "..." if len(llm_analysis.get("explanation", "")) > 80 else llm_analysis.get("explanation", "Social Media Post Scan")
    
    # Save to database
    db_result = ScanResult(
        title=f"AI Web Search: {title}",
        url=url,
        text_content=f"[Content retrieved via AI Web Search Grounding]\n\n{report_text[:1200]}...",
        ml_score=ml_score,
        llm_score=llm_score,
        trust_score=trust_score,
        verdict=verdict,
        bias_rating=llm_analysis["biasRating"],
        clickbait_score=int(llm_analysis["clickbaitScore"]),
        key_findings=json.dumps(llm_analysis["keyFindings"]),
        claims_analysed=json.dumps(llm_analysis["claimsAnalysed"])
    )
    
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    
    return db_result

