"""
API Test Script for AgriTech AI Backend
Tests all endpoints with sample data
"""

import requests
import json
from pathlib import Path
import sys

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"🧪 {title}")
    print("=" * 70)

def print_subsection(title):
    """Print formatted subsection header"""
    print(f"\n📝 {title}")
    print("-" * 70)

def test_health_check():
    """Test health check endpoint"""
    print_section("TEST 1: HEALTH CHECK")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:")
            print(json.dumps(data, indent=2))
            
            if data.get("status") == "healthy":
                print("\n✅ Health check PASSED!")
                print(f"   • Crop model loaded: {data.get('crop_model_loaded')}")
                print(f"   • Disease model loaded: {data.get('disease_model_loaded')}")
                return True
            else:
                print("\n⚠️  API running but some models not loaded")
                return False
        else:
            print(f"\n❌ Health check FAILED with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API server")
        print("\n💡 Make sure the server is running:")
        print("   1. Open a new terminal")
        print("   2. Navigate to backend folder")
        print("   3. Activate venv: source venv/bin/activate")
        print("   4. Run: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_crop_recommendation():
    """Test crop recommendation endpoint"""
    print_section("TEST 2: CROP RECOMMENDATION")
    
    # Test cases with different soil/weather conditions
    test_cases = [
        {
            "name": "Rice-friendly conditions (High rainfall, warm)",
            "data": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.87,
                "humidity": 82.00,
                "ph": 6.50,
                "rainfall": 202.93
            }
        },
        {
            "name": "Wheat-friendly conditions (Moderate climate)",
            "data": {
                "N": 70,
                "P": 60,
                "K": 40,
                "temperature": 20.0,
                "humidity": 60.0,
                "ph": 6.8,
                "rainfall": 80.0
            }
        },
        {
            "name": "Cotton-friendly conditions (Hot, moderate rain)",
            "data": {
                "N": 120,
                "P": 40,
                "K": 30,
                "temperature": 30.0,
                "humidity": 70.0,
                "ph": 7.0,
                "rainfall": 100.0
            }
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/recommend-crop",
                json=test_case['data'],
                timeout=10
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ SUCCESS!")
                print(f"   Recommended Crop: {result['recommended_crop']}")
                print(f"   Confidence: {result['confidence']:.2f}%")
                print(f"   Input: N={test_case['data']['N']}, P={test_case['data']['P']}, "
                      f"K={test_case['data']['K']}, Temp={test_case['data']['temperature']}°C")
                passed += 1
            else:
                print(f"❌ FAILED: {response.json()}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Crop Recommendation Tests: {passed} passed, {failed} failed")
    return failed == 0

def test_disease_detection(image_path=None):
    """Test disease detection endpoint"""
    print_section("TEST 3: DISEASE DETECTION")
    
    if image_path is None:
        print("⚠️  No image provided for testing")
        print("\n💡 To test disease detection:")
        print("   1. Download a sample plant leaf image")
        print("   2. Save it in the backend folder")
        print("   3. Run: python test_api.py --image <filename.jpg>")
        print("\n📥 Sample test images:")
        print("   • https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("   • Or use any plant leaf image (JPG/PNG)")
        return None
    
    image_file = Path(image_path)
    
    if not image_file.exists():
        print(f"❌ Image not found: {image_path}")
        print(f"\n💡 Make sure the image file exists in: {image_file.absolute()}")
        return False
    
    print(f"📸 Testing with image: {image_path}")
    print(f"📁 File size: {image_file.stat().st_size / 1024:.2f} KB")
    print("-" * 70)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_file.name, f, 'image/jpeg')}
            response = requests.post(
                f"{BASE_URL}/api/detect-disease",
                files=files,
                timeout=30
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"   Detected Disease: {result['disease']}")
            print(f"   Confidence: {result['confidence']:.2f}%")
            print(f"\n📊 Top 3 Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['disease']}")
                print(f"      Confidence: {pred['confidence']:.2f}%")
            return True
        else:
            print(f"❌ FAILED: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_crops():
    """Test get available crops endpoint"""
    print_section("TEST 4: GET AVAILABLE CROPS")
    
    try:
        response = requests.get(f"{BASE_URL}/api/crops", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"   Total crops available: {result['total']}")
            print(f"\n📋 Crop List:")
            for i, crop in enumerate(result['crops'], 1):
                print(f"   {i:2d}. {crop}")
            return True
        else:
            print(f"❌ FAILED: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_diseases():
    """Test get detectable diseases endpoint"""
    print_section("TEST 5: GET DETECTABLE DISEASES")
    
    try:
        response = requests.get(f"{BASE_URL}/api/diseases", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"   Total disease classes: {result['total']}")
            print(f"\n📋 Disease List (first 10):")
            for i, disease in enumerate(result['diseases'][:10], 1):
                print(f"   {i:2d}. {disease}")
            if result['total'] > 10:
                print(f"   ... and {result['total'] - 10} more diseases")
            return True
        else:
            print(f"❌ FAILED: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests(image_path=None):
    """Run all API tests"""
    print("\n" + "🌾 AGRITECH AI - API TESTING SUITE".center(70, "="))
    print(f"Server: {BASE_URL}")
    print("=" * 70)
    
    results = []
    
    # Test 1: Health Check
    result = test_health_check()
    results.append(("Health Check", result))
    
    if result is False:
        print("\n" + "=" * 70)
        print("❌ CRITICAL: Cannot connect to server or server not healthy")
        print("=" * 70)
        print("\n💡 Please start the server first:")
        print("   1. Open a new terminal")
        print("   2. cd backend")
        print("   3. source venv/bin/activate  (or venv\\Scripts\\activate on Windows)")
        print("   4. python main.py")
        print("\nThen run this test script again.")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    # Test 2: Crop Recommendation
    result = test_crop_recommendation()
    results.append(("Crop Recommendation", result))
    
    # Test 3: Disease Detection
    result = test_disease_detection(image_path)
    results.append(("Disease Detection", result if result is not None else "Skipped"))
    
    # Test 4: Get Crops
    result = test_get_crops()
    results.append(("Get Crops List", result))
    
    # Test 5: Get Diseases
    result = test_get_diseases()
    results.append(("Get Diseases List", result))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result == "Skipped")
    
    for test_name, result in results:
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"
        print(f"{status:12s} - {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)
    
    if failed == 0 and passed > 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your API is working perfectly!")
        print("\n🚀 Next steps:")
        print("   • Visit: http://localhost:8000/docs")
        print("   • Build a frontend")
        print("   • Add database integration")
        print("   • Deploy to production")
    elif failed > 0:
        print("\n⚠️  SOME TESTS FAILED")
        print("💡 Check the error messages above for details")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    import sys
    
    # Check for image argument
    image_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--image" and len(sys.argv) > 2:
            image_path = sys.argv[2]
        elif sys.argv[1] != "--help":
            image_path = sys.argv[1]
    
    if "--help" in sys.argv:
        print("\nAgriTech AI - API Test Script")
        print("=" * 70)
        print("\nUsage:")
        print("   python test_api.py                    # Test without image")
        print("   python test_api.py --image leaf.jpg   # Test with image")
        print("   python test_api.py leaf.jpg           # Test with image")
        print("\n" + "=" * 70 + "\n")
        sys.exit(0)
    
    run_all_tests(image_path)