"""
Model Loading Test Script
Tests if all model files exist and can be loaded correctly
"""

import os
import sys

print("\n" + "=" * 70)
print("🌾 AGRITECH AI - MODEL LOADING TEST")
print("=" * 70)

# Step 1: Check if model files exist
print("\n📁 STEP 1: Checking Model Files")
print("-" * 70)

required_files = {
    'Crop Model': 'models/crop_recommendation_model.pkl',
    'Scaler': 'models/scaler.pkl',
    'Disease Model': 'models/disease_detection_model.h5',
    'Class Indices': 'models/class_indices.json',
    'Class Labels': 'models/class_labels.json'
}

all_files_exist = True
total_size = 0

for name, filepath in required_files.items():
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    
    if exists:
        size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        total_size += size
        print(f"{status} {name:20s} : {filepath:45s} ({size:.2f} MB)")
    else:
        print(f"{status} {name:20s} : {filepath:45s} (MISSING)")
        all_files_exist = False

print(f"\n📊 Total size: {total_size:.2f} MB")

if not all_files_exist:
    print("\n" + "=" * 70)
    print("❌ SOME MODEL FILES ARE MISSING!")
    print("-" * 70)
    print("\n📥 To fix this:")
    print("   1. Go to your Kaggle notebooks")
    print("   2. Click 'Output' on the right sidebar")
    print("   3. Download these files:")
    print("      From Crop_Recommendation_System:")
    print("        • crop_recommendation_model.pkl")
    print("        • scaler.pkl")
    print("      From Crop_Disease_Detection_CNN:")
    print("        • disease_detection_model.h5")
    print("        • class_indices.json")
    print("        • class_labels.json")
    print("   4. Place them in the backend/models/ folder")
    print("=" * 70 + "\n")
    sys.exit(1)

# Step 2: Test loading models
print("\n📦 STEP 2: Testing Model Loading")
print("-" * 70)

errors = []

# Test Crop Recommendation Model
print("\n🌾 Crop Recommendation Model:")
try:
    import joblib
    import numpy as np
    
    crop_model = joblib.load('models/crop_recommendation_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    num_crops = len(crop_model.classes_)
    print(f"   ✅ Model loaded successfully")
    print(f"   ✅ Scaler loaded successfully")
    print(f"   📊 Number of crops: {num_crops}")
    print(f"   📋 Crops available: {', '.join(sorted(crop_model.classes_)[:5])}...")
    
    # Test prediction with sample data
    print(f"\n   🧪 Running test prediction...")
    test_input = np.array([[90, 42, 43, 20.87, 82.0, 6.5, 202.93]])
    test_scaled = scaler.transform(test_input)
    test_pred = crop_model.predict(test_scaled)[0]
    test_prob = crop_model.predict_proba(test_scaled)[0].max() * 100
    
    print(f"   ✅ Test input: N=90, P=42, K=43, Temp=20.87°C")
    print(f"   ✅ Prediction: {test_pred}")
    print(f"   ✅ Confidence: {test_prob:.1f}%")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    errors.append(("Crop Model", str(e)))

# Test Disease Detection Model
print("\n🦠 Disease Detection Model:")
try:
    import tensorflow as tf
    import json
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    print(f"   📥 Loading model...")
    disease_model = tf.keras.models.load_model('models/disease_detection_model.h5')
    
    print(f"   📥 Loading class labels...")
    with open('models/class_labels.json', 'r') as f:
        class_labels = json.load(f)
    
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    num_classes = len(class_labels)
    model_params = disease_model.count_params()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   ✅ Class labels loaded successfully")
    print(f"   ✅ Class indices loaded successfully")
    print(f"   📊 Number of disease classes: {num_classes}")
    print(f"   🔢 Model parameters: {model_params:,}")
    
    # Show sample diseases
    sample_diseases = list(class_labels.values())[:5]
    print(f"   📋 Sample diseases: {', '.join(sample_diseases)}...")
    
    # Test prediction with dummy image
    print(f"\n   🧪 Running test prediction...")
    test_image = tf.random.normal((1, 224, 224, 3))
    test_pred = disease_model.predict(test_image, verbose=0)
    
    print(f"   ✅ Test image size: 224x224x3")
    print(f"   ✅ Prediction shape: {test_pred.shape}")
    print(f"   ✅ Model accepts input correctly")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    errors.append(("Disease Model", str(e)))

# Summary
print("\n" + "=" * 70)

if errors:
    print("❌ SOME MODELS FAILED TO LOAD")
    print("-" * 70)
    for model_name, error in errors:
        print(f"\n⚠️  {model_name} Error:")
        print(f"   {error}")
    
    print("\n💡 Common fixes:")
    print("   1. Ensure TensorFlow version matches: 2.18.0")
    print("      Run: pip install tensorflow==2.18.0")
    print("   2. Ensure scikit-learn version matches: 1.2.2")
    print("      Run: pip install scikit-learn==1.2.2")
    print("   3. Re-download model files from Kaggle")
    print("   4. Check file integrity (not corrupted)")
    print("   5. Run: python verify_install.py")
    print("=" * 70 + "\n")
    sys.exit(1)
else:
    print("🎉 ALL MODELS LOADED SUCCESSFULLY!")
    print("-" * 70)
    print("✅ Crop Recommendation Model: Ready")
    print("✅ Disease Detection Model: Ready")
    print("✅ All support files: Ready")
    print("\n🚀 Next Steps:")
    print("   1. Start the API server:")
    print("      python main.py")
    print("   2. Visit: http://localhost:8000/docs")
    print("   3. Test the API:")
    print("      python test_api.py (in a new terminal)")
    print("=" * 70)

# Show file structure
print("\n📂 Your Model Directory Structure:")
print("-" * 70)
print("backend/")
print("└── models/")
for name, filepath in sorted(required_files.items()):
    filename = os.path.basename(filepath)
    size = os.path.getsize(filepath) / 1024  # KB
    print(f"    ├── {filename:35s} ({size:.1f} KB)")
print("=" * 70 + "\n")

# Exit successfully
sys.exit(0)