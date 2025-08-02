#!/usr/bin/env python3
"""
Model Verification Script
This script checks which ML model is configured and verifies it runs end-to-end.
Updated to support XGBoost model WITHOUT scaler, using only selected_features.pkl.
"""

import joblib
import os
from config import Config

def check_model_files():
    """Check if all required ML files exist"""
    print("Checking ML Model Files...")
    print("=" * 50)
    
    files_to_check = [
        (Config.MODEL_PATH, "Model File"),
        (Config.FEATURES_PATH, "Selected Features")
    ]
    
    all_files_exist = True
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"[OK] {description}: {file_path} ({file_size:,} bytes)")
        else:
            print(f"[ERROR] {description}: {file_path} - NOT FOUND")
            all_files_exist = False
    
    return all_files_exist

def verify_model():
    """Load and verify the model"""
    print("\nVerifying Model...")
    print("=" * 50)
    
    try:
        # Load model
        model = joblib.load(Config.MODEL_PATH)
        model_type = type(model).__name__
        print(f"[OK] Model Type: {model_type}")

        # Identify known families
        if 'XGB' in model_type or 'XGBoost' in model_type:
            print("[OK] XGBoost Regressor detected (no scaler expected)")
        elif 'RandomForest' in model_type:
            print("[OK] Random Forest Regressor detected")
        elif 'CatBoost' in model_type:
            print("[OK] CatBoost Regressor detected")
        else:
            print(f"[INFO] Detected model class: {model_type}")

        # Optional metadata
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                print(f"[OK] Model Parameters: {len(params)} parameters configured")
            if hasattr(model, 'n_features_in_'):
                print(f"[OK] Model expects feature count: {model.n_features_in_}")
        except Exception as e:
            print(f"[WARNING] Could not read model metadata: {e}")

        return model
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None

def verify_features():
    """Load and print selected features used by the app"""
    print("\nVerifying Selected Features...")
    print("=" * 50)
    try:
        features = joblib.load(Config.FEATURES_PATH)
        print(f"[OK] Features loaded from {Config.FEATURES_PATH}")
        print(f"[OK] Feature count: {len(features)}")
        print("[OK] Feature List (in order):")
        for i, feature in enumerate(features, 1):
            print(f"   {i}. {feature}")
        return features
    except Exception as e:
        print(f"[ERROR] Error loading selected features: {e}")
        return None

def test_prediction():
    """Test a sample prediction end-to-end using selected_features only (no scaler)"""
    print("\nTesting Sample Prediction...")
    print("=" * 50)
    try:
        import pandas as pd

        # Load model and features
        model = joblib.load(Config.MODEL_PATH)
        features = joblib.load(Config.FEATURES_PATH)

        # Build sample with all possible fields; filter to selected features only
        all_sample_features = {
            'year': 2020,
            'engineSize': 1.5,
            'tax': 150,
            'model': 5,  # This will be ignored if not in selected features
            'mileage': 12000,
            'mpg': 50.0,
            'transmission_Manual': 1
        }
        
        sample_df = pd.DataFrame([all_sample_features])
        print("[OK] Raw sample input constructed.")
        
        # Filter to only selected features (this excludes 'model' if not in features)
        sample_df = sample_df[features]
        print(f"[OK] Filtered to selected features: {list(sample_df.columns)}")
        print(f"[OK] Selected features from training: {features}")
        print(f"[OK] Sample values: {sample_df.values[0]}")

        # Direct predict (no scaling needed for XGBoost)
        y = model.predict(sample_df)[0]
        print(f"\n[OK] Prediction Result: ${float(y):,.2f}")
        print("[OK] XGBoost model inference path is working correctly without scaler.")
        return True

    except Exception as e:
        print(f"[ERROR] Error testing prediction: {e}")
        return False

def main():
    print("Car Price Prediction System - Model Verification")
    print("=" * 60)
    
    # Check if files exist
    if not check_model_files():
        print("\n[ERROR] Some ML files are missing!")
        print("Please ensure you have copied the following files to the project directory:")
        print(f"- XGBoost model file: {Config.MODEL_PATH}")
        print(f"- Selected features pickle: {Config.FEATURES_PATH}")
        return
    
    # Verify each component
    model = verify_model()
    features = verify_features()
    
    # Extra assertion that the configured model path points to the intended XGBoost artifact name
    configured_name = os.path.basename(Config.MODEL_PATH)
    if configured_name.lower() not in ("xgboost_model.pkl", "xboost_model.pkl"):
        print(f"\n[WARNING] MODEL_PATH is set to '{configured_name}', which is not named XGBoost_model.pkl or XBoost_model.pkl.")
        print("   This is only a naming check; the model type detection above is the source of truth.")
    
    if model and features:
        # Ensure model is indeed XGBoost
        model_type = type(model).__name__
        if not ('XGB' in model_type or 'XGBoost' in model_type):
            print("\n[ERROR] Verification failed: The loaded model is not XGBoost.")
            print(f"   Detected model: {model_type}")
            return
        else:
            print("[OK] Verified: Loaded model is XGBoost.")

        # Test prediction
        if test_prediction():
            print("\n[SUCCESS] All Model Components Verified Successfully!")
            print("[OK] XGBoost_model.pkl is connected correctly and ready for production!")
        else:
            print("\n[ERROR] Model test failed!")
    else:
        print("\n[ERROR] Model verification failed!")

if __name__ == "__main__":
    main()
