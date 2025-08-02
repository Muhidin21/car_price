#!/usr/bin/env python3
"""
Admin Setup Script for Car Prediction System
This script creates an admin user in the MongoDB database
"""

from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from datetime import datetime
import getpass

def create_admin_user():
    # MongoDB Atlas connection
    mongodb_uri = input("Enter your MongoDB Atlas connection string: ").strip()
    
    if not mongodb_uri:
        print("❌ MongoDB URI is required!")
        return False
    
    try:
        client = MongoClient(mongodb_uri)
        # Test the connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        
        db = client['car_prediction_db']
        users_collection = db['users']
    except Exception as e:
        print(f"❌ Error connecting to MongoDB Atlas: {e}")
        return False
    
    print("\n=== Car Prediction System - Admin Setup ===")
    print("This script will create an admin user for the system.\n")
    
    # Get admin details
    username = input("Enter admin username: ").strip()
    
    # Check if username already exists
    if users_collection.find_one({'username': username}):
        print(f"❌ Username '{username}' already exists!")
        return False
    
    email = input("Enter admin email: ").strip()
    
    # Get password securely
    while True:
        password = getpass.getpass("Enter admin password: ")
        confirm_password = getpass.getpass("Confirm admin password: ")
        
        if password == confirm_password:
            if len(password) < 6:
                print("❌ Password must be at least 6 characters long!")
                continue
            break
        else:
            print("❌ Passwords don't match! Please try again.")
    
    # Create admin user
    try:
        hashed_password = generate_password_hash(password)
        admin_data = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'role': 'admin',
            'created_at': datetime.now(),
            'created_by': 'admin_setup_script'
        }
        
        result = users_collection.insert_one(admin_data)
        
        if result.inserted_id:
            print(f"\n✅ Admin user '{username}' created successfully!")
            print(f"Admin ID: {result.inserted_id}")
            print("\nYou can now login to the system with these credentials.")
            return True
        else:
            print("❌ Failed to create admin user!")
            return False
            
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False
    
    finally:
        client.close()

def main():
    print("Starting admin setup...")
    print("This script will create an admin user in your MongoDB Atlas cluster.")
    print("\nTo get your connection string:")
    print("1. Go to MongoDB Atlas (https://cloud.mongodb.com)")
    print("2. Click 'Connect' on your cluster")
    print("3. Choose 'Connect your application'")
    print("4. Copy the connection string")
    print("5. Replace <password> with your actual password\n")
    
    success = create_admin_user()
    
    if success:
        print("\n🎉 Admin setup completed successfully!")
        print("You can now start the Flask application and login as admin.")
        print("\nDon't forget to set the MONGODB_URI environment variable in your Flask app!")
    else:
        print("\n❌ Admin setup failed!")

if __name__ == "__main__":
    main()