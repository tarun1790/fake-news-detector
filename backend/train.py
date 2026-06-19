import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

def load_and_normalize_datasets():
    dfs = []
    
    # ------------------ DATASET 1: GonzaloA/fake_news ------------------
    # Original mapping: 0 = Fake, 1 = Real
    # Target mapping: 0 = Real, 1 = Fake (Needs to be INVERTED!)
    print("Loading Dataset 1: GonzaloA/fake_news...")
    try:
        ds1 = load_dataset("GonzaloA/fake_news", split="train")
        df1 = pd.DataFrame(ds1)[["title", "text", "label"]]
        df1["label"] = df1["label"].astype(int)
        df1["label"] = 1 - df1["label"]  # Invert so 0=Real, 1=Fake
        print(f"-> Successfully loaded & inverted GonzaloA/fake_news: {len(df1)} rows.")
        dfs.append(df1)
    except Exception as e:
        print(f"-> Failed to load GonzaloA/fake_news: {e}")

    # ------------------ DATASET 2: ErfanMoosaviMonazzah/fake-news-detection-dataset-English ------------------
    # Original mapping: 0 = Fake, 1 = Real
    # Target mapping: 0 = Real, 1 = Fake (Needs to be INVERTED!)
    print("Loading Dataset 2: ErfanMoosaviMonazzah/fake-news-detection-dataset-English...")
    try:
        ds2 = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English", split="train")
        df2 = pd.DataFrame(ds2)[["title", "text", "label"]]
        df2 = df2.dropna(subset=["label"])
        df2["label"] = df2["label"].astype(int)
        df2["label"] = 1 - df2["label"]  # Invert so 0=Real, 1=Fake
        print(f"-> Successfully loaded & inverted ErfanMoosaviMonazzah: {len(df2)} rows.")
        dfs.append(df2)
    except Exception as e:
        print(f"-> Failed to load ErfanMoosaviMonazzah: {e}")

    # ------------------ FALLBACK BOOTSTRAP ------------------
    if len(dfs) == 0:
        print("All online datasets failed. Bootstrapping synthetic dataset...")
        data = {
            'title': [
                "US Election results are in", "Unbelievable: Aliens land in New York",
                "Scientists discover cure for common cold", "Government to ban all dogs next month",
                "Stocks hit record highs today", "Secret cure for cancer hidden by big pharma",
                "New park opens in downtown Seattle", "Local man wins lottery for the second time",
                "Drinking bleach cures viral infections, study claims", "NASA launches new satellite to Mars"
            ] * 200,
            'text': [
                "The election results have been certified and the winner announced officially today.",
                "Yesterday night, multiple UFOs were spotted landing in Central Park. Officials are silent.",
                "A breakthrough study from Oxford shows a molecule that completely cures the common cold virus.",
                "Leaked documents show the administration is drafting a bill to ban dog ownership nationwide.",
                "The S&P 500 and Dow Jones industrial average reached all-time high closing prices on Wednesday.",
                "An anonymous whistle-blower claims that a simple natural herb cures cancer but pharma blocks it.",
                "The city council inaugurated the new green space today with a public ribbon cutting ceremony.",
                "A local resident has defied all mathematical odds by winning the state lottery grand prize twice.",
                "A controversial online paper claims that drinking highly diluted chemical bleach cleanses viruses.",
                "The space agency has successfully launched its next generation rover to search for water on Mars."
            ] * 200,
            'label': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0] * 200
        }
        return pd.DataFrame(data)

    # Combine loaded dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["title"] = combined_df["title"].fillna("")
    combined_df["text"] = combined_df["text"].fillna("")
    return combined_df

def train_model():
    print("Starting Combined ML Model Training Pipeline...")
    
    # 1. Load Combined Data
    df = load_and_normalize_datasets()
    print(f"Merged Dataset Loaded. Total Combined Records: {len(df)}")
    
    # 2. Setup save paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(data_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(data_dir, "fake_news_model.joblib")

    # 3. Preprocess data
    df["full_text"] = df["title"] + " " + df["text"]
    
    # 4. Split Train/Test
    X = df["full_text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train/Test Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # 5. TF-IDF Vectorization
    print("Vectorizing combined text corpus...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # 6. Train Logistic Regression
    print("Training Logistic Regression classifier on aligned merged data...")
    model = LogisticRegression(max_iter=1500, C=1.0)
    model.fit(X_train_vectorized, y_train)
    
    # 7. Evaluate
    y_pred = model.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)
    print(f"Aligned Combined Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real News (0)", "Fake News (1)"]))
    
    # 8. Save
    print(f"Saving vectorizer: {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saving model: {model_path}")
    joblib.dump(model, model_path)
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    train_model()
