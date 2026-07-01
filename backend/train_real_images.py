import os
import io
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features_from_image(image_path):
    """Extract ELA, FFT, and YCbCr color features from a single image file."""
    try:
        with open(image_path, 'rb') as f:
            file_bytes = f.read()
            
        has_editing_software = 0
        keywords = [b'photoshop', b'gimp', b'adobe', b'canva', b'corel', b'paint.net', b'fotor', b'pixlr']
        for kw in keywords:
            if kw in file_bytes.lower():
                has_editing_software = 1
                break
                
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        has_exif = 0
        try:
            exif = image.getexif()
            if exif and len(exif) > 0:
                has_exif = 1
        except Exception:
            pass
            
        # ELA
        temp_io = io.BytesIO()
        image.save(temp_io, format='JPEG', quality=90)
        temp_io.seek(0)
        temp_img = Image.open(temp_io)
        diff = ImageChops.difference(image, temp_img)
        diff_arr = np.array(diff)
        ela_mean = float(np.mean(diff_arr))
        ela_std = float(np.std(diff_arr))
        ela_max = float(np.max(diff_arr))
        
        # FFT
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
        except Exception:
            pass
            
        # Color stats
        color_std_y = 50.0
        color_std_cb = 15.0
        color_std_cr = 15.0
        try:
            ycbcr = image.convert('YCbCr')
            y_arr, cb_arr, cr_arr = ycbcr.split()
            color_std_y = float(np.std(np.array(y_arr)))
            color_std_cb = float(np.std(np.array(cb_arr)))
            color_std_cr = float(np.std(np.array(cr_arr)))
        except Exception:
            pass
            
        return {
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
    except Exception as e:
        # print(f"Error processing {image_path}: {e}")
        return None

def build_real_world_dataset():
    real_dir = r"C:\Users\tarun\Downloads\archive\RealArt\RealArt"
    ai_dir = r"C:\Users\tarun\Downloads\archive\AiArtData\AiArtData"
    
    rows = []
    
    # Process Real Art (Class 0: Authentic)
    print("Extracting features from RealArt images...")
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    processed_count = 0
    for f in real_files:
        feats = extract_features_from_image(f)
        if feats:
            feats["label"] = 0
            rows.append(feats)
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(real_files)} RealArt images...")
            
    # Process AI Art (Class 2: AI-Generated)
    print("Extracting features from AiArtData images...")
    ai_files = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    processed_count = 0
    for f in ai_files:
        feats = extract_features_from_image(f)
        if feats:
            feats["label"] = 2
            rows.append(feats)
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(ai_files)} AiArtData images...")
            
    # Generate simulated Deepfake data (Class 1) to retain 3-class architecture support
    print("Generating simulated Deepfake baseline features...")
    n_deepfakes = 500
    for _ in range(n_deepfakes):
        # Deepfakes have elevated ELA due to localized blending, and no camera EXIF
        ela_mean = np.random.uniform(2.5, 7.5)
        ela_std = np.random.uniform(1.5, 5.0)
        ela_max = np.random.uniform(50.0, 95.0)
        fft_noise = np.random.uniform(0.10, 0.58)
        color_std_y = np.random.uniform(10.0, 80.0)
        color_std_cb = np.random.uniform(7.0, 32.0)
        color_std_cr = np.random.uniform(7.0, 32.0)
        rows.append({
            "ela_mean": ela_mean,
            "ela_std": ela_std,
            "ela_max": ela_max,
            "has_exif": 0.0,
            "has_editing_software": 0.0,
            "fft_high_freq_mean": fft_noise,
            "color_std_y": color_std_y,
            "color_std_cb": color_std_cb,
            "color_std_cr": color_std_cr,
            "label": 1
        })
        
    return pd.DataFrame(rows)

def train_classifier():
    df = build_real_world_dataset()
    print(f"Dataset compiled. Total samples: {len(df)}")
    print(df["label"].value_counts())
    
    X = df.drop(columns=["label"])
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier on real-world extracted features...")
    model = RandomForestClassifier(n_estimators=250, max_depth=16, random_state=42)
    model.fit(X_train.values, y_train)
    
    y_pred = model.predict(X_test.values)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Validation Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Authentic", "Deepfake", "AI-Generated"]))
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    model_path = os.path.join(data_dir, "fake_image_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved real-world trained model to: {model_path}")

if __name__ == "__main__":
    train_classifier()
