from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(url, key)

def save_crop_prediction(data, result):
    """Save crop prediction to database"""
    try:
        response = supabase.table('crop_predictions').insert({
            'user_id': None,  # Will add authentication later
            'nitrogen': float(data['N']),
            'phosphorus': float(data['P']),
            'potassium': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall']),
            'recommended_crop': result['recommended_crop'],
            'confidence': float(result['confidence'])
        }).execute()
        return response
    except Exception as e:
        print(f"Error saving crop prediction: {e}")
        return None

def save_disease_prediction(result):
    """Save disease prediction to database"""
    try:
        response = supabase.table('disease_predictions').insert({
            'user_id': None,  # Will add authentication later
            'detected_disease': result['disease'],
            'confidence': float(result['confidence']),
            'top_predictions': result['top_predictions']
        }).execute()
        return response
    except Exception as e:
        print(f"Error saving disease prediction: {e}")
        return None

def get_recent_crop_predictions(limit=10):
    """Get recent crop predictions"""
    try:
        response = supabase.table('crop_predictions')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching crop predictions: {e}")
        return []

def get_recent_disease_predictions(limit=10):
    """Get recent disease predictions"""
    try:
        response = supabase.table('disease_predictions')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching disease predictions: {e}")
        return []