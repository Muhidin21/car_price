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
    
    # Model Configuration - Using Random Forest
    MODEL_PATH = 'random_forest_model.pkl'
    SCALER_PATH = 'scaler.pkl'
    FEATURES_PATH = 'selected_features.pkl'
    
    # Email Configuration for OTP
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')
    
    # OTP Configuration
    OTP_EXPIRY_MINUTES = 10
    
    # Email Domain Restrictions
    ALLOWED_EMAIL_DOMAINS = ['gmail.com']  # Only allow Gmail addresses
    
    @staticmethod
    def validate_config():
        """Validate required configuration"""
        if not Config.MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is required")
        
        if Config.MONGODB_URI == 'mongodb+srv://<username>:<password>@<cluster-name>.mongodb.net/car_prediction_db?retryWrites=true&w=majority':
            raise ValueError("Please update MONGODB_URI with your actual MongoDB Atlas connection string")
        
        return True