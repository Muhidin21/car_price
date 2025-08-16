import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB Atlas Configuration
    MONGODB_URI = os.getenv('MONGODB_URI')
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database Configuration
    DATABASE_NAME = 'car_prediction_db'
    
    # Collections
    USERS_COLLECTION = 'users'
    PREDICTIONS_COLLECTION = 'predictions'
    
    # Model Configuration - default artifacts (update as needed)
    # Available hypertuned models saved by notebook:
    #   best_random_forest_model.pkl, best_xgboost_model.pkl, best_gradient_boosting_model.pkl,
    #   best_decision_tree_model.pkl, best_catboost_model.pkl
    # Baselines: hist_gradient_boosting_model.pkl, linear_regression_model.pkl
    # Default to the requested Random Forest model
    MODEL_PATH = os.getenv('MODEL_PATH', 'best_random_forest_model.pkl')
    SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.pkl')
    FEATURES_PATH = os.getenv('FEATURES_PATH', 'selected_features.pkl')
    
    # Email Configuration for OTP
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')
    
    # OTP Configuration
    # Increased to reduce timeouts during verification
    OTP_EXPIRY_MINUTES = 120
    
    # Email Domain Restrictions
    ALLOWED_EMAIL_DOMAINS = ['gmail.com']  # Only allow Gmail addresses
    
    # Email Validation Configuration
    ENABLE_MX_CHECK = os.getenv('ENABLE_MX_CHECK', 'False').lower() == 'true'  # Enable DNS MX record checking
    
    @staticmethod
    def validate_config():
        """Validate required configuration"""
        if not Config.MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is required")
        
        if Config.MONGODB_URI == 'mongodb+srv://<username>:<password>@<cluster-name>.mongodb.net/car_prediction_db?retryWrites=true&w=majority':
            raise ValueError("Please update MONGODB_URI with your actual MongoDB Atlas connection string")

        # Validate model artifacts exist (warn only; app.py will also handle)
        for p in [Config.MODEL_PATH, Config.SCALER_PATH, Config.FEATURES_PATH]:
            if not os.path.exists(p):
                print(f"[WARN] Missing artifact: {p} (ensure correct file in project root or set env path)")
            else:
                print(f"[OK] Found artifact: {p}")
        
        return True