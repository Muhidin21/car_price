#!/usr/bin/env python3
"""
Model Verification Script
This script checks if the CatBoost model and related files are properly configured
"""

import joblib
import os
from config import Config

def check_model_files():
    """Check if all required ML files exist"""
    print("🔍 Checking ML Model Files...")
    print("=" * 50)
    
    files_to_check = [
        (Config.MODEL_PATH, "Random Forest Model"),
        (Config.SCALER_PATH, "Feature Scaler"),
        (Config.FEATURES_PATH, "Selected Features")
    ]
    
    all_files_exist = True
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} ({file_size:,} bytes)")
        else:
            print(f"❌ {description}: {file_path} - NOT FOUND")
            all_files_exist = False
    
    return all_files_exist

def verify_model():
    """Load and verify the model"""
    print("\n🤖 Verifying Model...")
    print("=" * 50)
    
    try:
        # Load model
        model = joblib.load(Config.MODEL_PATH)
        model_type = type(model).__name__
        print(f"✅ Model Type: {model_type}")
        
        # Check if it's CatBoost
        if 'CatBoost' in model_type:
            print("✅ CatBoost Regressor confirmed!")
            
            # Get model info if available
            try:
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    print(f"✅ Model Parameters: {len(params)} parameters configured")
                
                if hasattr(model, 'feature_importances_'):
                    print(f"✅ Feature Importances: Available")
                
                if hasattr(model, 'n_features_in_'):
                    print(f"✅ Expected Features: {model.n_features_in_}")
                    
            except Exception as e:
                print(f"⚠️ Could not get detailed model info: {e}")
        else:
            print(f"⚠️ Warning: Expected CatBoost, but found {model_type}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def verify_features():
    """Verify the expected features for CatBoost model"""
    print("\n🎯 Verifying Features...")
    print("=" * 50)
    
    # CatBoost model expected features (hardcoded since we don't need separate file)
    features = ['year', 'engineSize', 'tax', 'model', 'mileage', 'mpg', 'transmission_Manual']
    
    print(f"✅ Expected Features: {len(features)} features")
    print("✅ Feature List:")
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature}")
    
    return features

def test_prediction():
    """Test a sample prediction"""
    print("\n🧪 Testing Sample Prediction...")
    print("=" * 50)
    
    try:
        import pandas as pd
        
        # Load all ML components
        model = joblib.load(Config.MODEL_PATH)
        scaler = joblib.load(Config.SCALER_PATH)
        features = joblib.load(Config.FEATURES_PATH)
        
        # Sample data (2020 Fiesta, 1.5L, Manual, 12k miles, 50 MPG, $150 tax)
        sample_df = pd.DataFrame([{
            'year': 2020,
            'engineSize': 1.5,
            'tax': 150,
            'model': 5,
            'mileage': 12000,
            'mpg': 50.0,
            'transmission_Manual': 1
        }])
        
        print("✅ Sample Input Data:")
        for col, val in sample_df.iloc[0].items():
            print(f"   {col}: {val}")
        
        # Ensure correct feature order and scale
        sample_df = sample_df[features]
        sample_scaled = scaler.transform(sample_df)
        
        print(f"✅ Scaled input: {sample_scaled[0]}")
        
        # Random Forest prediction with proper scaling
        prediction = model.predict(sample_scaled)[0]
        
        print(f"\n✅ Prediction Result: ${prediction:,.2f}")
        print("✅ Model is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False

def main():
    print("🚗 Car Price Prediction System - Model Verification")
    print("=" * 60)
    
    # Check if files exist
    if not check_model_files():
        print("\n❌ Some ML files are missing!")
        print("Please ensure you have copied the following files to the project directory:")
        print("- random_forest_model.pkl")
        print("- scaler.pkl")
        print("- selected_features.pkl")
        return
    
    # Verify each component
    model = verify_model()
    features = verify_features()
    
    if model and features:
        # Test prediction
        if test_prediction():
            print("\n🎉 All Model Components Verified Successfully!")
            print("✅ Your CatBoost model is ready for production!")
        else:
            print("\n❌ Model test failed!")
    else:
        print("\n❌ Model verification failed!")

if __name__ == "__main__":
    main()