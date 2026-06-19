import os
import re
import joblib
import numpy as np

class MLModelPredictor:
    def __init__(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        self.vectorizer_path = os.path.join(self.data_dir, "tfidf_vectorizer.joblib")
        self.model_path = os.path.join(self.data_dir, "fake_news_model.joblib")
        
        self.vectorizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Attempts to load the TF-IDF vectorizer and Logistic Regression model from disk."""
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.model_path):
            try:
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.model = joblib.load(self.model_path)
                print("Linguistic ML Model loaded successfully.")
            except Exception as e:
                print(f"Error loading saved ML model: {e}. Falling back to heuristics.")
                self.vectorizer = None
                self.model = None
        else:
            print("Linguistic ML Model files not found. Fallback heuristics will be used.")

    def predict(self, title: str, text: str) -> float:
        """
        Predicts the linguistic credibility score (0 - 100).
        100 means highly likely to be structurally/linguistically credible.
        0 means highly likely to be clickbait/fake linguistic style.
        """
        # Ensure model is re-checked in case it was trained after startup
        if self.model is None or self.vectorizer is None:
            self.load_model()
            
        full_text = f"{title or ''} {text or ''}"

        if self.model is not None and self.vectorizer is not None:
            try:
                # Transform text
                text_vectorized = self.vectorizer.transform([full_text])
                # predict_proba returns [prob_class_0, prob_class_1]
                # In train.py: label 0 = True/Reliable, label 1 = Fake/Unreliable
                probs = self.model.predict_proba(text_vectorized)[0]
                # Return percentage of reliable class (index 0)
                score = float(probs[0] * 100)
                return max(0.0, min(100.0, score))
            except Exception as e:
                print(f"Error running model prediction: {e}. Falling back to heuristics.")
        
        # Fallback linguistic heuristics (linguistic stylistic check)
        return self._heuristic_predict(title or "", text)

    def _heuristic_predict(self, title: str, text: str) -> float:
        """
        Fallback heuristic styling analyzer. Evaluates:
        - CAPS LOCK ratio (signals sensationalism)
        - Exclamation mark count (signals sensationalism)
        - Clickbait / sensational word frequencies
        """
        score = 80.0 # Start with neutral/slightly positive baseline

        # Check title
        if title:
            # 1. Capitalization in title
            caps_words = sum(1 for w in title.split() if w.isupper() and len(w) > 1)
            total_words = len(title.split()) or 1
            caps_ratio = caps_words / total_words
            if caps_ratio > 0.3:
                score -= 15.0
            
            # 2. Clickbait words in title
            sensational_words = ["unbelievable", "shocking", "secret", "reveal", "miracle", "exposed", "conspiracy", "illuminati", "aliens", "government hide"]
            title_lower = title.lower()
            for word in sensational_words:
                if word in title_lower:
                    score -= 8.0
            
            # 3. Exclamation marks
            if "!" in title:
                score -= 5.0
                if title.count("!") > 1:
                    score -= 5.0

        # Check body text
        if text:
            # Caps ratio in body
            body_caps = sum(1 for w in text.split()[:100] if w.isupper() and len(w) > 2) # check first 100 words
            body_total = len(text.split()[:100]) or 1
            body_caps_ratio = body_caps / body_total
            if body_caps_ratio > 0.15:
                score -= 10.0

            # Exclamation marks density in body
            excl_density = text.count("!") / (len(text.split()) or 1)
            if excl_density > 0.02:
                score -= 8.0
                
            # Suspicious sensational keywords density
            sensational_keywords = ["must share", "they don't want you to know", "proof", "miraculous", "banned", "classified"]
            text_lower = text.lower()
            keyword_count = sum(1 for w in sensational_keywords if w in text_lower)
            score -= (keyword_count * 4.0)

        # Bound score between 5.0 and 95.0 for heuristics
        return max(5.0, min(95.0, score))

# Global predictor instance
predictor = MLModelPredictor()
