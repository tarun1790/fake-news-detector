import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_dataset_features():
    """
    Simulate features representing 10 different image datasets with custom distributions:
    Classes:
    - 0: Authentic (Real, untouched)
    - 1: Deepfake (AI face manipulations, Celeb-DF, FF++, DFDC, WildDeepfake)
    - 2: AI-Generated (Entirely synthetic, CIFAKE, Artifact, GenImage, Auto-GAN)
    - 3: Morphed/Edited (Spliced, copy-pasted, or modified)
    """
    np.random.seed(42)
    n_samples_per_dataset = 600
    
    datasets = [
        # Real Datasets
        {"name": "FFHQ", "class": 0, "has_exif_prob": 0.45, "ela_mean_range": (1.0, 3.0), "ela_std_range": (0.5, 1.5), "fft_noise_range": (0.01, 0.05), "software_prob": 0.02},
        {"name": "LFW", "class": 0, "has_exif_prob": 0.35, "ela_mean_range": (0.8, 2.5), "ela_std_range": (0.4, 1.2), "fft_noise_range": (0.01, 0.04), "software_prob": 0.01},
        
        # Deepfake Datasets (Class 1)
        {"name": "Celeb-DF", "class": 1, "has_exif_prob": 0.02, "ela_mean_range": (8.0, 16.0), "ela_std_range": (4.0, 9.0), "fft_noise_range": (0.25, 0.55), "software_prob": 0.01},
        {"name": "FaceForensics++", "class": 1, "has_exif_prob": 0.01, "ela_mean_range": (7.0, 14.0), "ela_std_range": (3.5, 8.0), "fft_noise_range": (0.20, 0.50), "software_prob": 0.01},
        {"name": "DFDC", "class": 1, "has_exif_prob": 0.03, "ela_mean_range": (9.0, 18.0), "ela_std_range": (4.5, 10.0), "fft_noise_range": (0.28, 0.60), "software_prob": 0.01},
        {"name": "WildDeepfake", "class": 1, "has_exif_prob": 0.05, "ela_mean_range": (7.5, 15.0), "ela_std_range": (4.0, 8.5), "fft_noise_range": (0.22, 0.52), "software_prob": 0.02},
        
        # AI-Generated Datasets (Class 2)
        {"name": "CIFAKE", "class": 2, "has_exif_prob": 0.01, "ela_mean_range": (4.0, 8.0), "ela_std_range": (2.0, 4.5), "fft_noise_range": (0.45, 0.85), "software_prob": 0.01},
        {"name": "Artifact", "class": 2, "has_exif_prob": 0.02, "ela_mean_range": (4.5, 9.0), "ela_std_range": (2.2, 5.0), "fft_noise_range": (0.40, 0.80), "software_prob": 0.01},
        {"name": "GenImage", "class": 2, "has_exif_prob": 0.01, "ela_mean_range": (5.0, 10.0), "ela_std_range": (2.5, 5.5), "fft_noise_range": (0.50, 0.90), "software_prob": 0.01},
        {"name": "Auto-GAN", "class": 2, "has_exif_prob": 0.01, "ela_mean_range": (4.2, 8.5), "ela_std_range": (2.1, 4.8), "fft_noise_range": (0.42, 0.82), "software_prob": 0.01},
    ]
    
    rows = []
    for ds in datasets:
        for _ in range(n_samples_per_dataset):
            # Generate EXIF data presence
            has_exif = 1 if np.random.rand() < ds["has_exif_prob"] else 0
            
            # Generate Editing software marker presence
            has_software = 1 if np.random.rand() < ds["software_prob"] else 0
            
            # Generate ELA metrics
            ela_mean = np.random.uniform(*ds["ela_mean_range"])
            ela_std = np.random.uniform(*ds["ela_std_range"])
            ela_max = ela_mean + (ela_std * np.random.uniform(2.5, 4.5))
            
            # Generate FFT frequency artifacts
            fft_noise = np.random.uniform(*ds["fft_noise_range"])
            
            # Generate color statistics
            color_std_y = np.random.uniform(35.0, 75.0)
            if ds["class"] == 2: # AI Gen often has color channel correlation shifts
                color_std_cb = np.random.uniform(5.0, 12.0)
                color_std_cr = np.random.uniform(5.0, 12.0)
            else:
                color_std_cb = np.random.uniform(8.0, 22.0)
                color_std_cr = np.random.uniform(8.0, 22.0)
                
            rows.append({
                "ela_mean": ela_mean,
                "ela_std": ela_std,
                "ela_max": ela_max,
                "has_exif": has_exif,
                "has_editing_software": has_software,
                "fft_high_freq_mean": fft_noise,
                "color_std_y": color_std_y,
                "color_std_cb": color_std_cb,
                "color_std_cr": color_std_cr,
                "label": ds["class"]
            })
            
    # Add a synthetic Morphed/Edited dataset (Class 3)
    # Typically has high ELA discrepancy (very high max ELA, moderate mean), high software probability, camera EXIF often missing
    for _ in range(n_samples_per_dataset * 2): # Double the samples to balance the 4 classes
        has_exif = 1 if np.random.rand() < 0.15 else 0
        has_software = 1 if np.random.rand() < 0.75 else 0 # High probability of editing software markers
        ela_mean = np.random.uniform(6.0, 12.0)
        ela_std = np.random.uniform(5.0, 10.0)
        ela_max = np.random.uniform(80.0, 255.0) # Massive local ELA difference at boundaries
        fft_noise = np.random.uniform(0.05, 0.20)
        color_std_y = np.random.uniform(35.0, 75.0)
        color_std_cb = np.random.uniform(8.0, 22.0)
        color_std_cr = np.random.uniform(8.0, 22.0)
        
        rows.append({
            "ela_mean": ela_mean,
            "ela_std": ela_std,
            "ela_max": ela_max,
            "has_exif": has_exif,
            "has_editing_software": has_software,
            "fft_high_freq_mean": fft_noise,
            "color_std_y": color_std_y,
            "color_std_cb": color_std_cb,
            "color_std_cr": color_std_cr,
            "label": 3
        })
        
    return pd.DataFrame(rows)

def train_image_classifier():
    print("Initializing Image Forensic Classifier Training Pipeline...")
    
    # 1. Generate feature sets from 10 dataset profiles + morphed profile
    df = generate_dataset_features()
    print(f"Features generated successfully. Total records: {len(df)}")
    
    # 2. Split train/test
    X = df.drop(columns=["label"])
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 3. Train Random Forest Classifier
    print("Training Random Forest Classifier on image forensics features...")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Image Classifier Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[
        "Authentic (0)", "Deepfake (1)", "AI-Generated (2)", "Morphed/Edited (3)"
    ]))
    
    # 5. Save model
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    model_path = os.path.join(data_dir, "fake_image_model.joblib")
    
    joblib.dump(model, model_path)
    print(f"Model saved successfully to: {model_path}")
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    train_image_classifier()
