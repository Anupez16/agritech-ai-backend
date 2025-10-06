"""
AgriTech AI - FastAPI Backend
Crop Recommendation & Disease Detection System
Version: 1.0.0
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase_client import save_crop_prediction, save_disease_prediction
import numpy as np
import tensorflow as tf
import joblib
import json
from PIL import Image
import io
from typing import Dict, List, Any
import uvicorn
import os

# ===================================================================
# FASTAPI APP INITIALIZATION
# ===================================================================

app = FastAPI(
    title="AgriTech AI API",
    description="AI-powered Crop Recommendation & Plant Disease Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://agritech-ai-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# LOAD MODELS AT STARTUP
# ===================================================================

print("=" * 70)
print("🌾 AGRITECH AI - LOADING MODELS")
print("=" * 70)

# Global variables for models
crop_model = None
scaler = None
disease_model = None
class_labels = None
class_indices = None

# Load Crop Recommendation Model
try:
    crop_model = joblib.load('models/crop_recommendation_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Crop recommendation model loaded")
    print(f"   📊 Number of crops: {len(crop_model.classes_)}")
except Exception as e:
    print(f"❌ Error loading crop model: {e}")

# Load Disease Detection Model
try:
    disease_model = tf.keras.models.load_model('models/disease_detection_model.h5')
    
    with open('models/class_labels.json', 'r') as f:
        class_labels = json.load(f)
    
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    print("✅ Disease detection model loaded")
    print(f"   📊 Number of disease classes: {len(class_labels)}")
except Exception as e:
    print(f"❌ Error loading disease model: {e}")

print("=" * 70)

# ===================================================================
# REQUEST/RESPONSE MODELS
# ===================================================================

class CropRecommendationInput(BaseModel):
    """Input model for crop recommendation"""
    N: float = Field(..., description="Nitrogen content in soil (kg/ha)", ge=0, le=200)
    P: float = Field(..., description="Phosphorus content in soil (kg/ha)", ge=0, le=200)
    K: float = Field(..., description="Potassium content in soil (kg/ha)", ge=0, le=200)
    temperature: float = Field(..., description="Temperature in Celsius", ge=-10, le=60)
    humidity: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    ph: float = Field(..., description="Soil pH value", ge=0, le=14)
    rainfall: float = Field(..., description="Rainfall in mm", ge=0, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.87,
                "humidity": 82.00,
                "ph": 6.50,
                "rainfall": 202.93
            }
        }


class CropRecommendationResponse(BaseModel):
    """Response model for crop recommendation"""
    success: bool
    recommended_crop: str
    confidence: float
    input_parameters: Dict
    

class DiseaseDetectionResponse(BaseModel):
    """Response model for disease detection"""
    success: bool
    disease: str
    confidence: float
    top_predictions: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    crop_model_loaded: bool
    disease_model_loaded: bool
    version: str
    message: str


class CropListResponse(BaseModel):
    """Response model for crop list"""
    success: bool
    crops: List[str]
    total: int


class DiseaseListResponse(BaseModel):
    """Response model for disease list"""
    success: bool
    diseases: List[str]
    total: int

# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.get("/", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns the status of the API and loaded models
    """
    return {
        "status": "healthy" if (crop_model and disease_model) else "partial",
        "crop_model_loaded": crop_model is not None,
        "disease_model_loaded": disease_model is not None,
        "version": "1.0.0",
        "message": "AgriTech AI API is running!"
    }


@app.post("/api/recommend-crop", response_model=CropRecommendationResponse, tags=["Crop Recommendation"])
async def recommend_crop(data: CropRecommendationInput):
    """
    Recommend the best crop based on soil and environmental parameters
    
    **Parameters:**
    - **N**: Nitrogen content in soil (0-200 kg/ha)
    - **P**: Phosphorus content in soil (0-200 kg/ha)
    - **K**: Potassium content in soil (0-200 kg/ha)
    - **temperature**: Temperature in Celsius
    - **humidity**: Relative humidity (0-100%)
    - **ph**: Soil pH value (0-14)
    - **rainfall**: Rainfall in mm
    
    **Returns:**
    - Recommended crop name
    - Confidence score (0-100%)
    - Input parameters used
    """
    
    if crop_model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Crop recommendation model not available. Please check if model files are loaded correctly."
        )
    
    try:
        # Prepare input features
        features = np.array([[
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]])
        
        # Scale features using the same scaler from training
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = crop_model.predict(features_scaled)[0]
        
        # Get confidence (probability)
        probabilities = crop_model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities) * 100)
        
        # Prepare response
        result = {
            "success": True,
            "recommended_crop": prediction,
            "confidence": round(confidence, 2),
            "input_parameters": data.dict()
        }
        
        # Save to database
        save_crop_prediction(data.dict(), {
            "recommended_crop": prediction,
            "confidence": confidence
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/api/detect-disease", response_model=DiseaseDetectionResponse, tags=["Disease Detection"])
async def detect_disease(file: UploadFile = File(...)):
    """
    Detect plant disease from an uploaded leaf image
    
    **Accepts:** JPG, JPEG, PNG images
    
    **Returns:**
    - Disease name
    - Confidence score (0-100%)
    - Top 3 predictions with confidence scores
    """
    
    if disease_model is None or class_labels is None:
        raise HTTPException(
            status_code=503,
            detail="Disease detection model not available. Please check if model files are loaded correctly."
        )
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG or PNG image."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize to model input size (224x224)
        image = image.resize((224, 224))
        
        # Convert to array and normalize (0-1 range)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = disease_model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx] * 100)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                "disease": class_labels[str(idx)],
                "confidence": round(float(predictions[0][idx] * 100), 2)
            }
            for idx in top_3_idx
        ]
        
        # Prepare response
        result = {
            "success": True,
            "disease": class_labels[str(predicted_idx)],
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions
        }
        
        # Save to database
        save_disease_prediction({
            "disease": class_labels[str(predicted_idx)],
            "confidence": confidence,
            "top_predictions": top_predictions
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Image processing error: {str(e)}"
        )


@app.get("/api/crops", response_model=CropListResponse, tags=["Information"])
async def get_available_crops():
    """
    Get list of all crops the model can recommend
    
    **Returns:**
    - List of all available crops
    - Total number of crops
    """
    if crop_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Crop model not available"
        )
    
    crops = sorted(crop_model.classes_.tolist())
    return {
        "success": True,
        "crops": crops,
        "total": len(crops)
    }


@app.get("/api/diseases", response_model=DiseaseListResponse, tags=["Information"])
async def get_detectable_diseases():
    """
    Get list of all diseases the model can detect
    
    **Returns:**
    - List of all detectable diseases
    - Total number of disease classes
    """
    if class_labels is None:
        raise HTTPException(
            status_code=503, 
            detail="Disease model not available"
        )
    
    diseases = sorted(class_labels.values())
    return {
        "success": True,
        "diseases": diseases,
        "total": len(diseases)
    }


# ===================================================================
# STARTUP EVENT
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "=" * 70)
    print("🚀 AgriTech AI API - READY!")
    print("=" * 70)
    print("📡 API Documentation: http://localhost:8000/docs")
    print("📊 Health Check: http://localhost:8000/")
    print("=" * 70 + "\n")


# ===================================================================
# RUN SERVER
# ===================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )