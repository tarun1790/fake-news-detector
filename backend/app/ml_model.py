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

class MLImagePredictor:
    def __init__(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        self.model_path = os.path.join(self.data_dir, "fake_image_model.joblib")
        
        self.model = None
        self.load_model()

    def load_model(self):
        """Attempts to load the image forensic model from disk."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Forensic Image ML Model loaded successfully.")
            except Exception as e:
                print(f"Error loading saved Image ML model: {e}.")
                self.model = None
        else:
            print("Forensic Image ML Model files not found. Fallback classification will be used.")

    def predict(self, features: dict) -> dict:
        """
        Takes a dict of extracted image features and runs prediction.
        Returns:
        - verdict: str
        - authenticity_score: float (0 - 100)
        - probabilities: list of floats
        """
        if self.model is None:
            self.load_model()
            
        feature_names = [
            "ela_mean", "ela_std", "ela_max", "has_exif", 
            "has_editing_software", "fft_high_freq_mean", 
            "color_std_y", "color_std_cb", "color_std_cr"
        ]
        
        # Prepare input vector
        input_data = []
        for name in feature_names:
            input_data.append(features.get(name, 0.0))
            
        if self.model is not None:
            try:
                # predict_proba expects a numpy array of shape (n_samples, n_features)
                probs = self.model.predict_proba(np.array([input_data]))[0]
                
                # Dynamic Forensic Override Rules to guarantee real-world correctness
                ela_mean = features.get("ela_mean", 0.0)
                ela_max = features.get("ela_max", 0.0)
                has_exif = features.get("has_exif", 0.0)
                has_software = features.get("has_editing_software", 0.0)
                fft_noise = features.get("fft_high_freq_mean", 0.0)
                
                # Rule A: Clean camera photos with hardware EXIF signatures
                # If an image has valid camera EXIF headers, no editing software, and no extreme splicing ELA anomalies, it is Authentic!
                if has_exif > 0.5 and has_software < 0.5 and ela_max < 55.0:
                    probs = np.array([0.96, 0.02, 0.01, 0.01])
                    
                # Rule B: Strict metadata software check or extreme splicing ELA discrepancy
                # If editing software signature is found in headers, or extreme local compression difference occurs, it is Morphed/Edited
                elif has_software > 0.5 or ela_max > 100.0:
                    probs = np.array([0.05, 0.10, 0.05, 0.80])
                    
                # Rule C: Extreme high frequency noise (periodic grid artifacts)
                # If the high-frequency FFT energy ratio is extremely high, classify as AI-Generated
                elif fft_noise > 0.75:
                    probs = np.array([0.05, 0.10, 0.80, 0.05])
                
                class_names = ["Authentic", "Deepfake", "AI-Generated", "Morphed/Edited"]
                max_idx = np.argmax(probs)
                verdict = class_names[max_idx]
                authenticity_score = float(probs[0] * 100)
                
                return {
                    "verdict": verdict,
                    "authenticity_score": max(0.0, min(100.0, authenticity_score)),
                    "probabilities": [float(p * 100) for p in probs]
                }
            except Exception as e:
                print(f"Error running Image ML model prediction: {e}.")
                
        # Fallback prediction based on simple heuristics
        return self._heuristic_predict(features)

    def _heuristic_predict(self, features: dict) -> dict:
        ela_mean = features.get("ela_mean", 0.0)
        ela_max = features.get("ela_max", 0.0)
        has_exif = features.get("has_exif", 0.0)
        has_software = features.get("has_editing_software", 0.0)
        fft_noise = features.get("fft_high_freq_mean", 0.0)
        
        probs = [0.0, 0.0, 0.0, 0.0]
        
        if has_software > 0.5 or ela_max > 85.0:
            # Morphed/Edited
            probs[3] = 0.80
            probs[0] = 0.20
            verdict = "Morphed/Edited"
        elif fft_noise > 0.70:
            # AI-Generated
            probs[2] = 0.85
            probs[0] = 0.15
            verdict = "AI-Generated"
        elif ela_mean > 5.0:
            # Deepfake
            probs[1] = 0.80
            probs[0] = 0.20
            verdict = "Deepfake"
        else:
            # Authentic
            probs[0] = 0.95
            probs[3] = 0.05
            verdict = "Authentic"
            
        return {
            "verdict": verdict,
            "authenticity_score": float(probs[0] * 100),
            "probabilities": [float(p * 100) for p in probs]
        }

# Global image predictor instance
image_predictor = MLImagePredictor()
