# Luffy AI - Hybrid Fake News Detector 🕵️‍♂️💻

Luffy AI is a high-performance, hybrid fake news detection platform consisting of an **Apple-style Web Dashboard** and a **Manifest V3 Chrome Browser Extension**. 

The system leverages a hybrid detection pipeline combining a locally trained **Machine Learning Stylistic Classifier** and deep semantic **Gemini AI Fact-Checking** with real-time Google Search grounding.

---

## 🚀 Key Features

* **Apple Pro Obsidian Theme**: A modern, high-contrast, pure pitch-black dashboard with responsive card highlights and smooth scroll-snapping layout animations.
* **Hybrid Classification Pipeline**:
  1. **Stylistic Classification (ML)**: Evaluates linguistic structures, sensationalism, and stylistic syntax using TF-IDF and Logistic Regression.
  2. **Semantic Verification (AI)**: Extracts specific claims from articles and fact-checks them against active Google Search indexes using Gemini.
* **Social Media Scraping Fallback**: Standard scraping blocks on social media sites (Facebook, X, etc.) are bypassed by using Gemini Google Search Grounding to fetch verified context on URLs.
* **Chrome Extension**: Scan the active webpage with a single click, fetching immediate credibility verdicts, clickbait scores, and claim-level details directly from your browser toolbar.
* **SQLite Database Logging**: Scan records are stored locally, letting you browse, search, filter, and audit detailed historical reports.

---

## 📊 Datasets & Machine Learning Model

The Local ML Classifier is trained on a merged corpus of public datasets loaded from Hugging Face:

1. **GonzaloA/fake_news** (24,353 records)
2. **ErfanMoosaviMonazzah/fake-news-detection-dataset-English** (30,000 records)

### Data Normalization
The labels in both datasets were normalized into a single unified format:
* **`0` = Reliable / Real**
* **`1` = Misleading / Fake**

### Model Performance
* **Vectorization**: TF-IDF Vectorizer with unigrams and bigrams.
* **Model**: Logistic Regression.
* **Test Accuracy**: **`98.19%`** accuracy achieved on the merged 54,000+ record test split.
* **Model Files**: Saved locally in `backend/data/` for instantaneous inference inside FastAPI endpoints:
  - `tfidf_vectorizer.joblib`
  - `fake_news_model.joblib`

---

## 🛠️ Project Structure

```
├── backend/            # Python FastAPI backend
│   ├── app/            # Application logic (routes, database, analyzer, ML model)
│   ├── data/           # Trained models (.joblib) and local SQLite database (.db)
│   ├── train.py        # ML training script utilizing Hugging Face datasets
│   └── requirements.txt# Python backend dependencies
├── frontend/           # Web Dashboard files
│   ├── index.html      # Dashboard markup with scroll-snap welcome page
│   ├── style.css       # Obsidian Dark CSS design system
│   └── app.js          # Chart rendering and API integrations
├── extension/          # Manifest V3 Chrome Extension
│   ├── manifest.json   # Chrome Extension configuration
│   ├── popup.html      # Extension layout
│   ├── popup.css       # Obsidian-themed styling
│   └── popup.js        # Tab scraper and backend connector
└── README.md           # Documentation
```

---

## ⚙️ Setup & Installation

### Step 1: Run the Backend API Server
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the development server:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   *The backend will boot up at `http://127.0.0.1:8000`.*

### Step 2: Open the Web Dashboard
Open your browser and navigate to `http://localhost:8000` (or `http://127.0.0.1:8000`) to view the dashboard. Scroll down to enter and explore.

### Step 3: Install the Chrome Extension
1. Open **Google Chrome** and go to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in top-right).
3. Click **Load unpacked** (top-left).
4. Select the `extension/` folder in this project repository.
5. The extension is now loaded! Pin it to your toolbar to start scanning articles.

### Step 4: Configure Gemini AI Fact-Checking
1. Go to the **Settings** tab in the Web Dashboard.
2. Enter your **Google Gemini API Key** (get one free from [Google AI Studio](https://aistudio.google.com/)).
3. Click **Save API Key**. The system will dynamically switch from heuristics mode to active claim fact-checking!
