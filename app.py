from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from flask_mail import Mail, Message
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
import string
from dotenv import load_dotenv
from bson import ObjectId

# Load environment variables
load_dotenv()

from config import Config

app = Flask(__name__)

# Validate configuration
try:
    Config.validate_config()
    app.secret_key = Config.SECRET_KEY
    print("[OK] Configuration validated successfully")
except ValueError as e:
    print(f"[ERROR] Configuration error: {e}")
    print("Please check your .env file and update the MongoDB URI")
    exit(1)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = Config.MAIL_SERVER
app.config['MAIL_PORT'] = Config.MAIL_PORT
app.config['MAIL_USE_TLS'] = Config.MAIL_USE_TLS
app.config['MAIL_USERNAME'] = Config.MAIL_USERNAME
app.config['MAIL_PASSWORD'] = Config.MAIL_PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = Config.MAIL_DEFAULT_SENDER

mail = Mail(app)

# MongoDB Atlas connection
try:
    client = MongoClient(Config.MONGODB_URI)
    # Test the connection
    client.admin.command('ping')
    print("[OK] Connected to MongoDB Atlas successfully")
    
    db = client[Config.DATABASE_NAME]
    users_collection = db[Config.USERS_COLLECTION]
    predictions_collection = db[Config.PREDICTIONS_COLLECTION]
    otp_collection = db['otp_codes']  # New collection for OTP codes
    
except Exception as e:
    print(f"[ERROR] Error connecting to MongoDB Atlas: {e}")
    print("Please check your MongoDB Atlas connection string and network access")
    client = None
    db = None

# Load XGBoost model and features (no scaler needed)
try:
    # Load XGBoost model
    model = joblib.load(Config.MODEL_PATH)
    model_type = type(model).__name__
    print(f"[OK] XGBoost Model loaded successfully: {model_type}")
    
    # XGBoost does not require scaler
    scaler = None
    print("[INFO] Skipping scaler load (XGBoost model does not require scaling)")
    
    # Load selected features (order used at training)
    selected_features = joblib.load(Config.FEATURES_PATH)
    print(f"[OK] Features loaded successfully: {selected_features}")
    print(f"[OK] Number of features: {len(selected_features)}")
    
    if 'XGB' in model_type or 'XGBoost' in model_type:
        print("[OK] XGBoost Regressor detected")
        print("[OK] System ready for direct predictions without scaling")
    elif 'RandomForest' in model_type:
        print("[OK] Random Forest Regressor confirmed")
        print("[OK] System ready for predictions")
    elif 'CatBoost' in model_type:
        print("[OK] CatBoost Regressor detected")
        print("[OK] System ready for predictions")
    else:
        print(f"[OK] Model type: {model_type}")
        print("[OK] System ready for predictions")
    
    print("[OK] Complete ML pipeline loaded: Model + Features (no scaler needed)")

except Exception as e:
    print(f"[ERROR] Error loading ML components: {e}")
    print("Please make sure these files are in the project directory:")
    print("- XGBoost_model.pkl")
    print("- selected_features.pkl")
    model = None
    scaler = None
    selected_features = None

# OTP utility functions
def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email, otp, username):
    """Send OTP via email"""
    try:
        msg = Message(
            subject='Car Price Prediction - Email Verification',
            recipients=[email],
            html=f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0;">🚗 Car Price Prediction</h1>
                    <p style="color: white; margin: 10px 0 0 0;">Email Verification Required</p>
                </div>
                
                <div style="padding: 30px; background: #f8f9fa;">
                    <h2 style="color: #333;">Hello {username}!</h2>
                    <p style="color: #666; font-size: 16px;">
                        Thank you for registering with Car Price Prediction System. 
                        To complete your registration, please verify your email address.
                    </p>
                    
                    <div style="background: white; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                        <p style="color: #333; margin: 0 0 10px 0;">Your verification code is:</p>
                        <h1 style="color: #667eea; font-size: 32px; letter-spacing: 5px; margin: 0;">{otp}</h1>
                    </div>
                    
                    <p style="color: #666; font-size: 14px;">
                        This code will expire in 10 minutes. If you didn't request this verification, 
                        please ignore this email.
                    </p>
                    
                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                        <p style="color: #999; font-size: 12px; text-align: center;">
                            Car Price Prediction System<br>
                            Powered by Machine Learning
                        </p>
                    </div>
                </div>
            </div>
            '''
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def store_otp(email, otp):
    """Store OTP in database with expiry"""
    try:
        # Remove any existing OTP for this email
        otp_collection.delete_many({'email': email})
        
        # Store new OTP
        otp_data = {
            'email': email,
            'otp': otp,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(minutes=Config.OTP_EXPIRY_MINUTES),
            'verified': False
        }
        otp_collection.insert_one(otp_data)
        return True
    except Exception as e:
        print(f"Error storing OTP: {e}")
        return False

def verify_otp(email, otp):
    """Verify OTP code"""
    try:
        otp_record = otp_collection.find_one({
            'email': email,
            'otp': otp,
            'verified': False,
            'expires_at': {'$gt': datetime.now()}
        })
        
        if otp_record:
            # Mark OTP as verified
            otp_collection.update_one(
                {'_id': otp_record['_id']},
                {'$set': {'verified': True}}
            )
            return True
        return False
    except Exception as e:
        print(f"Error verifying OTP: {e}")
        return False

def is_email_domain_allowed(email):
    """Check if email domain is in the allowed list"""
    email_domain = email.lower().split('@')[-1]
    return email_domain in Config.ALLOWED_EMAIL_DOMAINS

def validate_email_format(email):
    """Validate email format and domain"""
    import re
    
    # Basic email format validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    
    # Check allowed domains
    if not is_email_domain_allowed(email):
        allowed_domains = ', '.join(f"@{domain}" for domain in Config.ALLOWED_EMAIL_DOMAINS)
        return False, f"Only {allowed_domains} addresses are allowed"
    
    return True, "Valid email"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['role'] = user['role']
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists')
            return render_template('register.html')
        
        # Check if email exists
        if users_collection.find_one({'email': email}):
            flash('Email already registered')
            return render_template('register.html')
        
        # Validate email format and domain
        is_valid, error_message = validate_email_format(email)
        if not is_valid:
            flash(error_message)
            return render_template('register.html')
        
        # Generate and send OTP
        otp = generate_otp()
        
        if send_otp_email(email, otp, username) and store_otp(email, otp):
            # Store user data temporarily in session
            session['temp_user_data'] = {
                'username': username,
                'email': email,
                'password': generate_password_hash(password)
            }
            
            flash('Please check your email for the verification code.')
            return redirect(url_for('verify_email'))
        else:
            flash('Error sending verification email. Please try again.')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    if 'temp_user_data' not in session:
        flash('Registration session expired. Please register again.')
        return redirect(url_for('register'))
    
    if request.method == 'POST':
        otp = request.form['otp']
        email = session['temp_user_data']['email']
        
        if verify_otp(email, otp):
            # Create the user account
            user_data = {
                'username': session['temp_user_data']['username'],
                'email': session['temp_user_data']['email'],
                'password': session['temp_user_data']['password'],
                'role': 'user',
                'email_verified': True,
                'created_at': datetime.now()
            }
            
            users_collection.insert_one(user_data)
            
            # Clear temporary data
            session.pop('temp_user_data', None)
            
            flash('Email verified successfully! You can now login.')
            return redirect(url_for('login'))
        else:
            flash('Invalid or expired verification code. Please try again.')
    
    return render_template('verify_email.html', email=session['temp_user_data']['email'])

@app.route('/resend_otp', methods=['POST'])
def resend_otp():
    if 'temp_user_data' not in session:
        return jsonify({'success': False, 'message': 'Registration session expired'})
    
    email = session['temp_user_data']['email']
    username = session['temp_user_data']['username']
    
    # Generate new OTP
    otp = generate_otp()
    
    if send_otp_email(email, otp, username) and store_otp(email, otp):
        return jsonify({'success': True, 'message': 'New verification code sent!'})
    else:
        return jsonify({'success': False, 'message': 'Error sending verification code'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    
    # Get user's prediction history
    user_predictions = list(predictions_collection.find({'user_id': session['user_id']}).sort('created_at', -1))
    
    return render_template('user_dashboard.html', predictions=user_predictions)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    # Get statistics
    total_users = users_collection.count_documents({'role': 'user'})
    total_predictions = predictions_collection.count_documents({})
    recent_predictions = list(predictions_collection.find().sort('created_at', -1).limit(10))
    
    return render_template('admin_dashboard.html', 
                         total_users=total_users,
                         total_predictions=total_predictions,
                         recent_predictions=recent_predictions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Check if model is loaded (no scaler needed for XGBoost)
            if model is None:
                flash('Error: XGBoost model is not loaded. Please check server logs.')
                return render_template('predict.html')
            
            # Get and validate form data
            year = int(request.form['year'])
            engine_size = float(request.form['engine_size'])
            tax = float(request.form['tax'])
            mileage = int(request.form['mileage'])
            mpg = float(request.form['mpg'])
            transmission_manual = int(request.form['transmission_manual'])
            
            # Enhanced input validation with warnings
            warnings = []
            
            # Basic range validation
            if not (1990 <= year <= 2025):
                flash('Year must be between 1990 and 2025')
                return render_template('predict.html')
            
            if not (0.5 <= engine_size <= 8.0):
                flash('Engine size must be between 0.5 and 8.0 liters')
                return render_template('predict.html')
            
            if not (0 <= tax <= 1000):
                flash('Tax must be between 0 and 1000')
                return render_template('predict.html')
            
            if not (0 <= mileage <= 500000):
                flash('Mileage must be between 0 and 500,000')
                return render_template('predict.html')
            
            if not (10 <= mpg <= 100):
                flash('MPG must be between 10 and 100')
                return render_template('predict.html')
            
            if transmission_manual not in [0, 1]:
                flash('Invalid transmission type')
                return render_template('predict.html')
            
            # Realistic value warnings
            if engine_size > 3.0:
                warnings.append(f'Engine size {engine_size}L is very large (typical: 1.0-2.5L). This will increase the predicted price significantly.')
            
            if year >= 2018 and mpg < 30:
                warnings.append(f'MPG {mpg} seems low for a {year} car (modern cars typically: 35-60 MPG)')
            
            if tax < 100:
                warnings.append(f'Tax £{tax} seems low (typical: £150-£300)')
            
            # Show warnings to user
            for warning in warnings:
                flash(f'⚠️ Warning: {warning}', 'warning')
            
            # Prepare data for XGBoost prediction (no scaling needed)
            print(f"[DEBUG] Raw inputs: year={year}, engine={engine_size}, tax={tax}, mileage={mileage}, mpg={mpg}, trans={transmission_manual}")
            
            # Create DataFrame with only the features used by XGBoost model
            input_features_data = {
                'year': year,
                'engineSize': engine_size,
                'tax': tax,
                'mileage': mileage,
                'mpg': mpg,
                'transmission_Manual': transmission_manual
            }
            
            # Create DataFrame and ensure correct feature order from training
            input_data = pd.DataFrame([input_features_data])
            input_data = input_data[selected_features]
            print(f"[DEBUG] DataFrame input: {input_data.values[0]}")
            print(f"[DEBUG] Feature order: {list(input_data.columns)}")
            print(f"[DEBUG] Selected features from training: {selected_features}")
            
            # Make prediction with XGBoost (no scaling needed)
            predicted_price = model.predict(input_data)[0]
            # Convert numpy.float32 to Python float for MongoDB compatibility
            predicted_price = float(predicted_price)
            predicted_price = round(predicted_price, 2)
            
            print(f"[OK] Prediction Result: ${predicted_price}")  # Debug log
            
            # Validate prediction result
            if predicted_price <= 0:
                flash('Error: Invalid prediction result. Please check your input values.')
                return render_template('predict.html')
            
            if predicted_price > 1000000:  # Sanity check for extremely high prices
                flash('Warning: Predicted price seems unusually high. Please verify your inputs.')
                # Continue anyway, but show warning
            
            # Save prediction to database (no model_type since it's not used)
            prediction_data = {
                'user_id': session['user_id'],
                'username': session['username'],
                'year': year,
                'engine_size': engine_size,
                'tax': tax,
                'mileage': mileage,
                'mpg': mpg,
                'transmission_manual': transmission_manual,
                'predicted_price': predicted_price,
                'created_at': datetime.now()
            }
            
            predictions_collection.insert_one(prediction_data)
            
            return render_template('predict.html',
                                 predicted_price=predicted_price,
                                 form_data=request.form)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}')
            return render_template('predict.html')
    
    return render_template('predict.html')

# Admin routes for user management
@app.route('/admin/users')
def admin_get_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        # Get all users with their prediction counts
        # Because predictions.user_id is stored as string(str(users._id)) in this app,
        # a direct $lookup on localField:_id won't match (types differ). Use a pipeline
        # $lookup to convert _id to string for the join, then compute counts.
        pipeline = [
            {'$match': {'role': 'user'}},
            {'$lookup': {
                'from': 'predictions',
                'let': {'uid': {'$toString': '$_id'}},
                'pipeline': [
                    {'$match': {'$expr': {'$eq': ['$user_id', '$$uid']}}}
                ],
                'as': 'predictions'
            }},
            {'$addFields': {
                'prediction_count': {'$size': '$predictions'}
            }},
            {'$project': {
                'username': 1,
                'email': 1,
                'created_at': 1,
                'prediction_count': 1
            }},
            {'$sort': {'created_at': -1}}
        ]
        
        users = list(users_collection.aggregate(pipeline))
        
        # Convert ObjectId to string for JSON serialization
        for user in users:
            user['_id'] = str(user['_id'])
            user['created_at'] = user['created_at'].isoformat()
        
        return jsonify({'success': True, 'users': users})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/user/<user_id>')
def admin_get_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    try:
        from bson import ObjectId

        # Try to interpret as ObjectId for users._id. If it fails, return 400.
        try:
            oid = ObjectId(user_id)
        except Exception:
            return jsonify({'success': False, 'message': 'Invalid user id format'}), 400

        # Lookup user by ObjectId
        user_doc = users_collection.find_one({'_id': oid}, {'username': 1, 'email': 1, 'created_at': 1})
        if not user_doc:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # Count predictions where predictions.user_id stored as string of users._id
        # Ensure we count by the same type stored in predictions.user_id (session stores str(ObjectId))
        prediction_count = predictions_collection.count_documents({'user_id': str(oid)})

        user = {
            '_id': str(user_doc['_id']),
            'username': user_doc.get('username'),
            'email': user_doc.get('email'),
            'created_at': user_doc.get('created_at').isoformat() if user_doc.get('created_at') else None,
            'prediction_count': int(prediction_count)
        }

        return jsonify({'success': True, 'user': user})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/user/<user_id>', methods=['DELETE'])
def admin_delete_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    try:
        from bson import ObjectId

        # Validate ObjectId
        try:
            oid = ObjectId(user_id)
        except Exception:
            return jsonify({'success': False, 'message': 'Invalid user id format'}), 400

        # Don't allow deleting admin users
        user = users_collection.find_one({'_id': oid})
        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        if user.get('role') == 'admin':
            return jsonify({'success': False, 'message': 'Cannot delete admin users'}), 400

        # Delete user's predictions first (predictions.user_id stored as str of users._id)
        predictions_collection.delete_many({'user_id': str(oid)})

        # Delete the user
        result = users_collection.delete_one({'_id': oid})

        if result.deleted_count > 0:
            return jsonify({'success': True, 'message': 'User deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'User not found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/export/users')
def admin_export_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    try:
        import csv
        from io import StringIO
        from flask import make_response
        
        # Get all users with correct prediction counts (join by stringified _id)
        pipeline = [
            {'$match': {'role': 'user'}},
            {'$lookup': {
                'from': 'predictions',
                'let': {'uid': {'$toString': '$_id'}},
                'pipeline': [
                    {'$match': {'$expr': {'$eq': ['$user_id', '$$uid']}}}
                ],
                'as': 'predictions'
            }},
            {'$addFields': {
                'prediction_count': {'$size': '$predictions'}
            }},
            {'$project': {
                'username': 1,
                'email': 1,
                'created_at': 1,
                'prediction_count': 1
            }},
            {'$sort': {'created_at': -1}}
        ]
        
        users = list(users_collection.aggregate(pipeline))
        
        # Create CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Username', 'Email', 'Joined Date', 'Total Predictions'])
        
        # Write data
        for user in users:
            joined = user.get('created_at')
            if isinstance(joined, datetime):
                joined_str = joined.strftime('%Y-%m-%d')
            else:
                # In case it's already a string/isoformat from future changes
                joined_str = str(joined)[:10] if joined else ''
            writer.writerow([
                user.get('username', ''),
                user.get('email', ''),
                joined_str,
                user.get('prediction_count', 0)
            ])
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=users_export.csv'
        
        return response
        
    except Exception as e:
        flash(f'Error exporting users: {str(e)}')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/reports/summary')
def admin_reports_summary():
    """Return aggregated counts for reports (past 7 days by default)."""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=6)  # 7-day window including today
        # Build daily labels
        days = []
        cursor = start
        for i in range(7):
            days.append(cursor.strftime('%a'))  # Mon, Tue...
            cursor += timedelta(days=1)

        # Aggregate predictions per day
        pred_pipeline = [
            {'$match': {'created_at': {'$gte': start, '$lte': end}}},
            {'$group': {
                '_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}},
                'count': {'$sum': 1}
            }}
        ]
        pred_docs = list(predictions_collection.aggregate(pred_pipeline))
        pred_map = {d['_id']: d['count'] for d in pred_docs}

        # Aggregate new users per day
        user_pipeline = [
            {'$match': {'role': 'user', 'created_at': {'$gte': start, '$lte': end}}},
            {'$group': {
                '_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}},
                'count': {'$sum': 1}
            }}
        ]
        user_docs = list(users_collection.aggregate(user_pipeline))
        user_map = {d['_id']: d['count'] for d in user_docs}

        # Build series aligned to labels
        series_dates = []
        pred_series = []
        user_series = []
        cursor = start
        for i in range(7):
            key = cursor.strftime('%Y-%m-%d')
            series_dates.append(cursor.strftime('%a'))
            pred_series.append(int(pred_map.get(key, 0)))
            user_series.append(int(user_map.get(key, 0)))
            cursor += timedelta(days=1)

        return jsonify({
            'success': True,
            'labels': series_dates,
            'predictions': pred_series,
            'new_users': user_series
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/predictions')
def admin_list_predictions():
    """Return paginated list of all predictions with user info for admin."""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        # Basic pagination
        page = int(request.args.get('page', 1))
        page_size = min(max(int(request.args.get('page_size', 20)), 1), 100)
        skip = (page - 1) * page_size

        total = predictions_collection.count_documents({})
        cursor = predictions_collection.find().sort('created_at', -1).skip(skip).limit(page_size)
        items = []
        for p in cursor:
            items.append({
                '_id': str(p.get('_id')),
                'user_id': p.get('user_id'),
                'username': p.get('username'),
                'year': p.get('year'),
                'engine_size': p.get('engine_size'),
                'tax': p.get('tax'),
                'model': p.get('model'),
                'mileage': p.get('mileage'),
                'mpg': p.get('mpg'),
                'transmission': 'Manual' if p.get('transmission_manual', 0) == 1 else 'Automatic',
                'predicted_price': float(p.get('predicted_price', 0)),
                'created_at': p.get('created_at').isoformat() if p.get('created_at') else None
            })
        return jsonify({'success': True, 'total': total, 'page': page, 'page_size': page_size, 'items': items})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/predictions/<pred_id>', methods=['DELETE'])
def admin_delete_prediction(pred_id):
    """Delete a specific prediction by id."""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        from bson import ObjectId
        try:
            oid = ObjectId(pred_id)
        except Exception:
            return jsonify({'success': False, 'message': 'Invalid prediction id'}), 400

        result = predictions_collection.delete_one({'_id': oid})
        if result.deleted_count > 0:
            return jsonify({'success': True, 'message': 'Prediction deleted'})
        else:
            return jsonify({'success': False, 'message': 'Prediction not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/export/predictions')
def admin_export_predictions():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    try:
        import csv
        from io import StringIO
        from flask import make_response
        
        # Get all predictions (already stored with primitive types)
        predictions = list(predictions_collection.find().sort('created_at', -1))
        
        # Create CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Username', 'Date', 'Year', 'Engine Size', 'Tax', 'Model', 
            'Mileage', 'MPG', 'Transmission', 'Predicted Price'
        ])
        
        # Write data safely handling datetimes
        for p in predictions:
            created = p.get('created_at')
            if isinstance(created, datetime):
                created_str = created.strftime('%Y-%m-%d %H:%M')
            else:
                created_str = str(created) if created else ''
            writer.writerow([
                p.get('username', ''),
                created_str,
                p.get('year', ''),
                p.get('engine_size', ''),
                p.get('tax', ''),
                p.get('model', ''),
                p.get('mileage', ''),
                p.get('mpg', ''),
                'Manual' if p.get('transmission_manual', 0) == 1 else 'Automatic',
                f"${float(p.get('predicted_price', 0)):.2f}"
            ])
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=predictions_report.csv'
        
        return response
        
    except Exception as e:
        flash(f'Error exporting predictions: {str(e)}')
        return redirect(url_for('admin_dashboard'))

@app.route('/test_model')
def test_model():
    """Test route to verify model is working"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    try:
        if model is None:
            return jsonify({
                'success': False, 
                'message': 'Model is not loaded'
            })
        
        # Test prediction with sample data (2020 car)
        # Create all possible features first
        all_test_features = {
            'year': 2020,
            'engineSize': 1.5,
            'tax': 150,
            'model': 5,
            'mileage': 12000,
            'mpg': 50.0,
            'transmission_Manual': 1
        }
        
        test_df = pd.DataFrame([all_test_features])
        
        # Filter to only selected features (excludes 'model' if not in selected_features)
        test_df = test_df[selected_features]
        prediction = model.predict(test_df)[0]
        # Convert numpy.float32 to Python float for JSON serialization
        prediction = float(prediction)
        
        return jsonify({
            'success': True,
            'model_type': type(model).__name__,
            'selected_features': selected_features,
            'test_input_filtered': test_df.iloc[0].to_dict(),
            'predicted_price': round(prediction, 2),
            'message': 'XGBoost model is working correctly with selected features!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Model test failed: {str(e)}'
        })

@app.route('/profile')
def user_profile():
    """User profile page for viewing own info"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        oid = ObjectId(session['user_id'])
        user = users_collection.find_one({'_id': oid})
        if not user:
            flash('User not found')
            return redirect(url_for('login'))

        # Count predictions (stored with user_id as string of ObjectId)
        pred_count = predictions_collection.count_documents({'user_id': str(oid)})

        # Normalize data for template
        user_view = {
            'id': str(user['_id']),
            'username': user.get('username', ''),
            'email': user.get('email', ''),
            'role': user.get('role', 'user'),
            'created_at': user.get('created_at').strftime('%Y-%m-%d') if user.get('created_at') else ''
        }

        return render_template('user_profile.html', user=user_view, prediction_count=pred_count)
    except Exception as e:
        flash(f'Error loading profile: {e}')
        return redirect(url_for('home'))

@app.route('/admin/email_domains')
def admin_email_domains():
    """Admin route to view allowed email domains"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    return jsonify({
        'success': True,
        'allowed_domains': Config.ALLOWED_EMAIL_DOMAINS,
        'message': f"Currently allowing: {', '.join(f'@{domain}' for domain in Config.ALLOWED_EMAIL_DOMAINS)}"
    })

if __name__ == '__main__':
    app.run(debug=True)
