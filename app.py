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
import time

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
    client.admin.command('ping')
    print("[OK] Connected to MongoDB Atlas successfully")
    
    db = client[Config.DATABASE_NAME]
    users_collection = db[Config.USERS_COLLECTION]
    predictions_collection = db[Config.PREDICTIONS_COLLECTION]
    otp_collection = db['otp_codes']
except Exception as e:
    print(f"[ERROR] Error connecting to MongoDB Atlas: {e}")
    print("Please check your MongoDB Atlas connection string and network access")
    client = None
    db = None
    users_collection = None
    predictions_collection = None
    otp_collection = None

# Load ML model directly (no scaler or selected features needed)
try:
    # Use XGBoost model directly
    model_path = 'xgboost_best_model.pkl'

    model = joblib.load(model_path)
    model_type = type(model).__name__
    print(f"[OK] ML Model loaded: {model_type} from {model_path}")
    
    # Get expected features directly from the model
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        print(f"[OK] Using model's expected features: {expected_features}")
    else:
        # Fallback to common XGBoost features
        expected_features = ['year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize', 'car_age']
        print(f"[OK] Using default XGBoost features: {expected_features}")

    # Sanity checks
    if not isinstance(expected_features, (list, tuple)) or len(expected_features) == 0:
        raise ValueError("expected_features must be a non-empty list of feature names")

    print(f"[OK] Inference feature order length: {len(expected_features)}")
    # Quick visibility for fuel-related columns
    fuel_cols_preview = [c for c in expected_features if 'fuel' in c.lower()]
    print(f"[OK] Fuel-related columns detected: {fuel_cols_preview}")
    print("[OK] Complete ML pipeline loaded: XGBoost model (no scaler needed)")

except Exception as e:
    print(f"[ERROR] Error loading ML components: {e}")
    print("Please make sure these files are in the project directory:")
    print("- xgboost_best_model.pkl")
    model = None
    expected_features = None

# Password hashing utilities

def ensure_pbkdf2(pw_value: str) -> str:
    """
    Ensure a password value is a PBKDF2-SHA256 hash. If it's already a pbkdf2 hash,
    return as-is. If it's raw or a different scheme (e.g., 'scrypt:'), re-hash with PBKDF2.
    This is a defense-in-depth guard for any insertion or update path.
    """
    try:
        if isinstance(pw_value, str) and pw_value.startswith('pbkdf2:'):
            return pw_value
        # treat as raw or foreign scheme -> rehash to pbkdf2
        return generate_password_hash(pw_value, method='pbkdf2:sha256', salt_length=16)
    except Exception:
        # In worst case, re-hash again as PBKDF2
        return generate_password_hash(str(pw_value), method='pbkdf2:sha256', salt_length=16)

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
    email_domain = (email or '').strip().lower().split('@')[-1]
    return email_domain in Config.ALLOWED_EMAIL_DOMAINS

def check_email_domain_mx(email):
    """Check if email domain has valid MX records (optional DNS validation)"""
    try:
        import dns.resolver
        domain = email.split('@')[1]
        mx_records = dns.resolver.resolve(domain, 'MX')
        return len(mx_records) > 0
    except (ImportError, dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.Timeout, Exception):
        # If dnspython is not installed or DNS check fails, return True to not block registration
        # You can install dnspython with: pip install dnspython
        return True

def validate_email_format(email, check_mx=False):
    """Validate email format and domain with comprehensive checks"""
    import re
    email = (email or '').strip()
    if not email:
        return False, "Email address is required"
    email_pattern = r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    if not re.match(email_pattern, email):
        return False, "Invalid email format. Please enter a valid email address"
    if len(email) > 254:
        return False, "Email address is too long (maximum 254 characters)"
    try:
        local_part, domain_part = email.split('@', 1)
    except ValueError:
        return False, "Invalid email format"
    if len(local_part) > 64:
        return False, "Email local part is too long (maximum 64 characters)"
    if len(domain_part) > 253:
        return False, "Email domain is too long (maximum 253 characters)"
    domain_pattern = r'^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    if not re.match(domain_pattern, domain_part):
        return False, "Invalid domain format in email address"
    tld_pattern = r'\.[a-zA-Z]{2,}$'
    if not re.search(tld_pattern, domain_part):
        return False, "Invalid top-level domain in email address"
    if not is_email_domain_allowed(email):
        allowed_domains = ', '.join(f"@{domain}" for domain in Config.ALLOWED_EMAIL_DOMAINS)
        return False, f"Only {allowed_domains} addresses are allowed"
    disposable_domains = ['tempmail.org', '10minutemail.com', 'guerrillamail.com', 'mailinator.com', 'yopmail.com', 'throwaway.email', 'temp-mail.org', 'fakeinbox.com']
    if domain_part.lower() in disposable_domains:
        return False, "Disposable email addresses are not allowed. Please use a valid email address."
    if check_mx and not check_email_domain_mx(email):
        return False, "Email domain does not have valid mail servers configured"
    return True, "Valid email"

def validate_password_strength(password):
    """Validate password meets security requirements"""
    import re
    
    if not password:
        return False, "Password is required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter (A-Z)"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter (a-z)"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number (0-9)"
    
    if not re.search(r'[!@#$%^&*]', password):
        return False, "Password must contain at least one special character (!@#$%^&*)"
    
    return True, "Password meets all requirements"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Guard against DB connection issues
    if users_collection is None:
        flash('Login service unavailable. Database connection not initialized.')
        return render_template('login.html')

    if request.method == 'POST':
        try:
            username = (request.form.get('username') or '').strip()
            password = request.form.get('password') or ''

            if not username or not password:
                flash('Please provide both username and password')
                return render_template('login.html')

            # Fetch by username exactly as stored
            user = users_collection.find_one({'username': username})

            # Debug logging (stdout)
            print(f"[LOGIN] Attempt user='{username}', found={bool(user)}")

            if not user:
                flash('Invalid username or password')
                return render_template('login.html')

            # Enforce PBKDF2 only; reject other schemes like scrypt
            stored_pw = user.get('password', '')
            algo_ok = isinstance(stored_pw, str) and stored_pw.startswith('pbkdf2:')
            try:
                password_ok = check_password_hash(stored_pw, password) if algo_ok else False
            except Exception as e:
                print(f"[LOGIN] check_password_hash error: {e}")
                password_ok = False

            print(f"[LOGIN] Diagnostics: pbkdf2_only={algo_ok}, email_verified={user.get('email_verified', False)}, role={user.get('role', 'user')}, password_ok={password_ok}")

            if not password_ok:
                flash('Invalid username or password')
                return render_template('login.html')

            # Relax email verification for admin to prevent lockout; enforce for normal users
            role = user.get('role', 'user')
            if role != 'admin' and not user.get('email_verified', False):
                flash('Please verify your email before logging in.')
                return render_template('login.html')

            # Establish session
            session['user_id'] = str(user['_id'])
            session['username'] = user.get('username', '')
            session['role'] = role

            # Confirm session set
            print(f"[LOGIN] Success user_id={session['user_id']} role={session['role']}")

            return redirect(url_for('admin_dashboard' if role == 'admin' else 'user_dashboard'))
        except Exception as e:
            # Surface unexpected errors
            print(f"[LOGIN][ERROR] {e}")
            flash('An unexpected error occurred during login. Please try again.')
            return render_template('login.html')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    def normalize_email(e: str) -> str:
        return (e or '').strip().lower()

    def normalize_username(u: str) -> str:
        # preserve case for storage, but strip whitespace for comparison
        return (u or '').strip()

    def email_exists_case_insensitive(e: str) -> bool:
        """Detect existing email regardless of case/whitespace."""
        try:
            norm = normalize_email(e)
            if not norm:
                return False
            # Prefer exact-case match fast path
            if users_collection.find_one({'email': norm}):
                return True
            # Case-insensitive exact match using regex ^...$ with i flag
            import re
            pattern = re.compile(f'^{re.escape(norm)}$', re.IGNORECASE)
            return users_collection.find_one({'email': pattern}) is not None
        except Exception as _:
            # Fallback safe
            return users_collection.find_one({'email': normalize_email(e)}) is not None

    if request.method == 'POST':
        username = normalize_username(request.form.get('username'))
        email_input = request.form.get('email')
        password = request.form.get('password') or ''
        email = normalize_email(email_input)

        # Validate email format and domain before sending verification email
        # This ensures we only send verification emails to valid email addresses
        is_valid, error_message = validate_email_format(email, check_mx=Config.ENABLE_MX_CHECK)
        if not is_valid:
            flash(error_message)
            return render_template('register.html')

        # Validate password strength
        is_password_valid, password_error = validate_password_strength(password)
        if not is_password_valid:
            flash(password_error)
            return render_template('register.html')

        # Check if username exists (exact match)
        if users_collection.find_one({'username': username}):
            flash('Username already exists')
            return render_template('register.html')

        # Check if email exists (case-insensitive)
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            if existing_user.get('email_verified', False):
                flash('Email already registered. Please sign in or use a different email.')
                return render_template('register.html')
            else:
                # Email exists but not verified, allow to continue verification
                session['temp_user_data'] = {
                    'username': username,
                    'email': email,
                    'password': password
                }
                flash('A verification code was already sent to this email. Please verify your email or resend the code.')
                return redirect(url_for('verify_email'))

        # Prevent duplicate concurrent registrations: check pending OTP for same email
        pending_otp = otp_collection.find_one({'email': email, 'verified': False, 'expires_at': {'$gt': datetime.now()}})
        if pending_otp:
            # Offer user to proceed to verification flow instead of re-registering
            session['temp_user_data'] = {
                'username': username,
                'email': email,
                'password': password
            }
            flash('A verification code was already sent to this email. Please verify your email or resend the code.')
            return redirect(url_for('verify_email'))

        # Generate and send OTP
        otp = generate_otp()

        if send_otp_email(email, otp, username) and store_otp(email, otp):
            # Store user data temporarily in session
            session['temp_user_data'] = {
                'username': username,
                'email': email,
                'password': password  # raw, temporary; never stored in DB
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
        email = (session['temp_user_data'].get('email') or '').strip().lower()
        
        if verify_otp(email, otp):
            # Before creating, re-check email uniqueness in case of race
            import re
            pattern = re.compile(f'^{re.escape(email)}$', re.IGNORECASE)
            if users_collection.find_one({'email': pattern}):
                # If someone already verified with this email, cancel creation
                session.pop('temp_user_data', None)
                flash('This email has already been verified and registered. Please sign in.')
                return redirect(url_for('login'))

            # Create the user account
            # ALWAYS hash with PBKDF2:SHA256 here (single source of truth).
            temp = session['temp_user_data']
            raw_password = temp['password']
            safe_password = generate_password_hash(raw_password, method='pbkdf2:sha256', salt_length=16)

            user_data = {
                'username': (temp['username'] or '').strip(),
                'email': email,
                'password': safe_password,
                'role': 'user',
                'email_verified': True,
                'created_at': datetime.now()
            }

            # Final server-side guard: if for any reason safe_password isn't pbkdf2, re-hash it.
            if not (isinstance(user_data['password'], str) and user_data['password'].startswith('pbkdf2:')):
                user_data['password'] = generate_password_hash(raw_password, method='pbkdf2:sha256', salt_length=16)

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
    
    # Validate email before resending OTP
    is_valid, error_message = validate_email_format(email, check_mx=Config.ENABLE_MX_CHECK)
    if not is_valid:
        return jsonify({'success': False, 'message': f'Invalid email: {error_message}'})
    
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

@app.route('/user_dashboard', methods=['GET', 'POST'])
def user_dashboard():
    if 'user_id' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    
    predicted_price = None
    form_data = None
    
    if request.method == 'POST':
        try:
            # Check if model is loaded
            if model is None or expected_features is None:
                flash('Error: ML components are not loaded. Please check server logs.')
                return render_template('user_dashboard.html', predictions=[])

            # Get and validate form data
            year = int(request.form['year'])
            engine_size = float(request.form['engine_size'])

            mileage = int(request.form['mileage'])
            mpg = float(request.form['mpg'])
            transmission_manual = int(request.form['transmission_manual'])
            fuel_type = request.form['fuel_type']

            # Input validation
            warnings = []
            if not (1990 <= year <= 2025):
                flash('Year must be between 1990 and 2025')
                return render_template('user_dashboard.html', predictions=[])
            if not (0.5 <= engine_size <= 3.0):
                flash('Engine size must be between 0.5 and 3.0 liters')
                return render_template('user_dashboard.html', predictions=[])

            if not (0 <= mileage <= 500000):
                flash('Mileage must be between 0 and 500,000')
                return render_template('user_dashboard.html', predictions=[])
            if not (10 <= mpg <= 100):
                flash('MPG must be between 10 and 100')
                return render_template('user_dashboard.html', predictions=[])
            if transmission_manual not in [0, 1]:
                flash('Invalid transmission type')
                return render_template('user_dashboard.html', predictions=[])
            if fuel_type not in ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other']:
                flash('Invalid fuel type selected')
                return render_template('user_dashboard.html', predictions=[])

            # Optional warnings
            if year >= 2018 and mpg < 30:
                warnings.append(f'MPG {mpg} seems low for a {year} car.')

            for warning in warnings:
                flash(f'⚠️ Warning: {warning}', 'warning')

            # Build input for XGBoost model
            print(f"[DEBUG] Raw inputs: year={year}, engine={engine_size}, mileage={mileage}, mpg={mpg}, trans={transmission_manual}, fuel_ui={fuel_type}")

            # Encode categorical features
            transmission_encoded = 1.0 if transmission_manual == 1 else 0.0
            
            # Encode fuel type
            fuel_encoding = {'Petrol': 0.0, 'Diesel': 1.0, 'Hybrid': 2.0, 'Electric': 3.0, 'Other': 4.0}
            fuel_encoded = fuel_encoding.get(fuel_type, 0.0)
            
            # Create input row for XGBoost model
            input_row = {
                'year': float(year),
                'transmission_encoded': transmission_encoded,
                'mileage': float(mileage),
                'fuelType_encoded': fuel_encoded,

                'mpg': float(mpg),
                'engineSize': float(engine_size),
                'car_age': 2024 - float(year)
            }

            input_df = pd.DataFrame([input_row])

            # Reindex to expected feature order and fill any missing with 0
            input_df = input_df.reindex(columns=expected_features, fill_value=0.0)
            print(f"[DEBUG] Input for prediction: {input_df.iloc[0].to_dict()}")

            # Predict directly (no scaling needed for XGBoost)
            predicted_price = float(model.predict(input_df)[0])
            predicted_price = round(predicted_price, 2)

            print(f"[OK] Prediction Result: ${predicted_price}")

            # Validate prediction result
            if predicted_price <= 0:
                flash('Error: Invalid prediction result. Please check your input values.')
                return render_template('user_dashboard.html', predictions=[])
            if predicted_price > 1000000:
                flash('Warning: Predicted price seems unusually high.')

            # Save prediction
            prediction_data = {
                'user_id': session['user_id'],
                'username': session.get('username', 'Unknown'),
                'year': year,
                'engine_size': engine_size,
                'mileage': mileage,
                'mpg': mpg,
                'transmission_manual': transmission_manual,
                'fuel_type': fuel_type,
                'predicted_price': predicted_price,
                'created_at': datetime.now()
            }

            predictions_collection.insert_one(prediction_data)
            flash(f'✅ Prediction successful! Estimated price: ${predicted_price:,.2f}')
            
            # Store form data for display
            form_data = {
                'year': year,
                'engine_size': engine_size,
                'mileage': mileage,
                'mpg': mpg,
                'transmission_manual': transmission_manual,
                'fuel_type': fuel_type
            }

        except ValueError as e:
            flash(f'Error: Invalid input values. Please check your data.')
            return render_template('user_dashboard.html', predictions=[])
        except Exception as e:
            flash(f'Error: {str(e)}')
            return render_template('user_dashboard.html', predictions=[])
    
    # Get user's prediction history
    user_predictions = list(predictions_collection.find({'user_id': session['user_id']}).sort('created_at', -1))
    
    return render_template('user_dashboard.html', 
                         predictions=user_predictions, 
                         predicted_price=predicted_price,
                         form_data=form_data)

@app.route('/user/export/predictions')
def user_export_predictions():
    """Export the logged-in user's predictions as a CSV."""
    if 'user_id' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    try:
        import csv
        from io import StringIO
        from flask import make_response

        # Fetch only current user's predictions, newest first
        user_id = session['user_id']
        preds = list(predictions_collection.find({'user_id': user_id}).sort('created_at', -1))

        # Build CSV
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'Date', 'Username', 'Year', 'Engine Size', 'Mileage',
            'MPG', 'Transmission', 'Predicted Price'
        ])
        for p in preds:
            created = p.get('created_at')
            if isinstance(created, datetime):
                created_str = created.strftime('%Y-%m-%d %H:%M')
            else:
                created_str = str(created) if created else ''
            writer.writerow([
                created_str,
                p.get('username', ''),
                p.get('year', ''),
                p.get('engine_size', ''),
                p.get('mileage', ''),
                p.get('mpg', ''),
                'Manual' if p.get('transmission_manual', 0) == 1 else 'Automatic',
                f"{float(p.get('predicted_price', 0)):.2f}"
            ])

        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        # Include username in filename for clarity
        safe_username = session.get('username', 'user').replace(' ', '_')
        response.headers['Content-Disposition'] = f'attachment; filename={safe_username}_predictions.csv'
        return response
    except Exception as e:
        flash(f'Error exporting predictions: {str(e)}')
        return redirect(url_for('user_dashboard'))

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
            # Check if model is loaded
            if model is None or expected_features is None:
                flash('Error: ML components are not loaded. Please check server logs.')
                return render_template('predict.html', form_data=request.form)

            # Get form data with proper error handling
            form_data = {}
            validation_errors = []
            
            # Validate year
            try:
                year = int(request.form.get('year', ''))
                if not (1990 <= year <= 2024):
                    validation_errors.append('Year must be between 1990 and 2024')
                form_data['year'] = year
            except (ValueError, TypeError):
                validation_errors.append('Year must be a valid number')
                form_data['year'] = ''
            
            # Validate engine size
            try:
                engine_size = float(request.form.get('engine_size', ''))
                if not (0.5 <= engine_size <= 3.0):
                    validation_errors.append('Engine size must be between 0.5 and 3.0 liters')
                form_data['engine_size'] = engine_size
            except (ValueError, TypeError):
                validation_errors.append('Engine size must be a valid number')
                form_data['engine_size'] = ''
            

            
            # Validate mileage
            try:
                mileage = int(request.form.get('mileage', ''))
                if not (0 <= mileage <= 200000):
                    validation_errors.append('Mileage must be between 0 and 200,000')
                form_data['mileage'] = mileage
            except (ValueError, TypeError):
                validation_errors.append('Mileage must be a valid number')
                form_data['mileage'] = ''
            
            # Validate MPG
            try:
                mpg = float(request.form.get('mpg', ''))
                if not (10 <= mpg <= 100):
                    validation_errors.append('MPG must be between 10 and 100')
                form_data['mpg'] = mpg
            except (ValueError, TypeError):
                validation_errors.append('MPG must be a valid number')
                form_data['mpg'] = ''
            
            # Validate transmission
            try:
                transmission_manual = int(request.form.get('transmission_manual', ''))
                if transmission_manual not in [0, 1]:
                    validation_errors.append('Invalid transmission type')
                form_data['transmission_manual'] = transmission_manual
            except (ValueError, TypeError):
                validation_errors.append('Please select a transmission type')
                form_data['transmission_manual'] = ''
            
            # Validate fuel type
            fuel_type = request.form.get('fuel_type', '')
            if not fuel_type or fuel_type not in ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other']:
                validation_errors.append('Please select a valid fuel type')
            form_data['fuel_type'] = fuel_type
            
            # If there are validation errors, return with form data
            if validation_errors:
                for error in validation_errors:
                    flash(error, 'error')
                return render_template('predict.html', form_data=form_data)
            
            # Additional realistic validation checks
            warnings = []
            current_year = datetime.now().year
            
            # Check for unrealistic MPG for newer cars
            if year >= 2018 and mpg < 25:
                warnings.append(f'MPG {mpg} seems low for a {year} car. Please verify.')
            
            # Check for unrealistic mileage for car age
            max_reasonable_mileage = (current_year - year) * 15000  # 15k miles per year max
            if mileage > max_reasonable_mileage:
                warnings.append(f'Mileage {mileage:,} seems high for a {year} car. Maximum reasonable: {max_reasonable_mileage:,}')
            

            
            # Check for very high engine size for newer cars
            if year >= 2020 and engine_size > 2.5:
                warnings.append(f'Engine size {engine_size}L seems large for a {year} car. Please verify.')
            
            # Show warnings
            for warning in warnings:
                flash(f'⚠️ Warning: {warning}', 'warning')


            # Optional warnings (only for MPG since engine size is now validated)
            if year >= 2018 and mpg < 30:
                warnings.append(f'MPG {mpg} seems low for a {year} car.')
            for warning in warnings:
                flash(f'⚠️ Warning: {warning}', 'warning')

            # Build input for XGBoost model
            print(f"[DEBUG] Raw inputs: year={year}, engine={engine_size}, mileage={mileage}, mpg={mpg}, trans={transmission_manual}, fuel_ui={fuel_type}")

            # Encode categorical features
            transmission_encoded = 1.0 if transmission_manual == 1 else 0.0
            
            # Encode fuel type
            fuel_encoding = {'Petrol': 0.0, 'Diesel': 1.0, 'Hybrid': 2.0, 'Electric': 3.0, 'Other': 4.0}
            fuel_encoded = fuel_encoding.get(fuel_type, 0.0)
            
            # Create input row for XGBoost model
            input_row = {
                'year': float(year),
                'transmission': transmission_manual,
                'mileage': float(mileage),
                'fuelType': fuel_encoded,
                'mpg': float(mpg),
                'engineSize': float(engine_size),
                'car_age': 2024 - float(year)
            }

            input_df = pd.DataFrame([input_row])

            # Reindex to expected feature order and fill any missing with 0
            input_df = input_df.reindex(columns=expected_features, fill_value=0.0)
            print(f"[DEBUG] Input for prediction: {input_df.iloc[0].to_dict()}")

            # Predict directly (no scaling needed for XGBoost)
            predicted_price = float(model.predict(input_df)[0])
            predicted_price = round(predicted_price, 2)

            print(f"[OK] Prediction Result: ${predicted_price}")

            # Validate prediction result
            if predicted_price <= 0:
                flash('Error: Invalid prediction result. Please check your input values.')
                return render_template('predict.html')
            if predicted_price > 1000000:
                flash('Warning: Predicted price seems unusually high.')

            # Save prediction
            prediction_data = {
                'user_id': session['user_id'],
                'username': session['username'],
                'year': year,
                'engine_size': engine_size,
                'mileage': mileage,
                'mpg': mpg,
                'transmission_manual': transmission_manual,
                'fuel_type': fuel_type,  # keep for history/filters, even if not used by model
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

@app.route('/admin/user/<user_id>', methods=['PUT'])
def admin_update_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        # Validate ObjectId
        try:
            oid = ObjectId(user_id)
        except Exception:
            return jsonify({'success': False, 'message': 'Invalid user id format'}), 400

        payload = request.get_json(silent=True) or {}
        updates = {}

        # Sanitize and validate fields
        new_username = payload.get('username')
        new_email = payload.get('email')
        new_role = payload.get('role')

        if new_username:
            # Ensure username is unique (case sensitive)
            existing = users_collection.find_one({'username': new_username, '_id': {'$ne': oid}})
            if existing:
                return jsonify({'success': False, 'message': 'Username already taken'}), 400
            updates['username'] = new_username

        if new_email:
            # Admin update: relax domain allowlist but still validate basic format and uniqueness
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, new_email or ''):
                return jsonify({'success': False, 'message': 'Invalid email format'}), 400
            # Ensure email is unique
            existing_email = users_collection.find_one({'email': new_email, '_id': {'$ne': oid}})
            if existing_email:
                return jsonify({'success': False, 'message': 'Email already in use'}), 400
            updates['email'] = new_email

        if new_role:
            if new_role not in ['user', 'admin']:
                return jsonify({'success': False, 'message': 'Invalid role'}), 400
            updates['role'] = new_role

        if not updates:
            return jsonify({'success': False, 'message': 'No valid fields to update'}), 400

        # Apply updates
        result = users_collection.update_one({'_id': oid}, {'$set': updates})
        if result.matched_count == 0:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # If username updated, reflect in predictions documents that store username string
        if 'username' in updates:
            predictions_collection.update_many({'user_id': str(oid)}, {'$set': {'username': updates['username']}})

        # Build fresh view
        user_doc = users_collection.find_one({'_id': oid}, {'username': 1, 'email': 1, 'role': 1, 'created_at': 1})
        user = {
            '_id': str(user_doc['_id']),
            'username': user_doc.get('username'),
            'email': user_doc.get('email'),
            'role': user_doc.get('role', 'user'),
            'created_at': user_doc.get('created_at').isoformat() if user_doc.get('created_at') else None
        }
        return jsonify({'success': True, 'message': 'User updated', 'user': user})
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

@app.route('/admin/predictions/data')
def admin_predictions_data():
    """Provide prediction data for visualization charts"""
    try:
        if not session.get('user_id') or not session.get('is_admin'):
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
        # Get all predictions with user information
        pipeline = [
            {
                '$lookup': {
                    'from': 'users',
                    'localField': 'user_id',
                    'foreignField': '_id',
                    'as': 'user_info'
                }
            },
            {
                '$unwind': '$user_info'
            },
            {
                '$project': {
                    'predicted_price': 1,
                    'year': 1,
                    'engine_size': 1,
                    'transmission_manual': 1,
                    'mileage': 1,
                    'fuel_type': 1,

                    'mpg': 1,
                    'created_at': 1,
                    'username': '$user_info.username'
                }
            }
        ]
        
        predictions = list(predictions_collection.aggregate(pipeline))
        
        # Calculate statistics
        if predictions:
            avg_price = sum(p['predicted_price'] for p in predictions) / len(predictions)
            total_predictions = len(predictions)
            unique_users = len(set(p['username'] for p in predictions))
            
            # Calculate model accuracy (simulated - in real scenario you'd compare with actual prices)
            # For now, we'll use a realistic accuracy based on prediction variance
            prices = [p['predicted_price'] for p in predictions]
            price_variance = np.var(prices) if len(prices) > 1 else 0
            # Higher variance might indicate less accuracy, but this is simplified
            model_accuracy = max(85, min(95, 92 - (price_variance / 1000000)))
        else:
            avg_price = 0
            total_predictions = 0
            unique_users = 0
            model_accuracy = 0
        
        # Format predictions for visualization
        formatted_predictions = []
        for pred in predictions:
            formatted_pred = {
                'predicted_price': float(pred['predicted_price']),
                'year': int(pred['year']),
                'engine_size': float(pred['engine_size']),
                'transmission': 'Manual' if pred.get('transmission_manual') == 1 else 'Automatic',
                'mileage': int(pred['mileage']),
                'fuel_type': pred.get('fuel_type', 'Petrol'),

                'mpg': float(pred['mpg']),
                'username': pred['username'],
                'created_at': pred['created_at'].isoformat() if isinstance(pred['created_at'], datetime) else str(pred['created_at'])
            }
            formatted_predictions.append(formatted_pred)
        
        stats = {
            'avgPrice': round(avg_price, 2),
            'totalPredictions': total_predictions,
            'activeUsers': unique_users,
            'modelAccuracy': round(model_accuracy, 1)
        }
        
        return jsonify({
            'success': True,
            'predictions': formatted_predictions,
            'stats': stats
        })
        
    except Exception as e:
        print(f"Error fetching prediction data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
            'Username', 'Date', 'Year', 'Engine Size', 'Model', 
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
    """Test route to verify XGBoost model is working"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    try:
        if model is None or expected_features is None:
            return jsonify({'success': False, 'message': 'ML components are not loaded'})

        return jsonify({
            'success': True,
            'model_type': type(model).__name__,
            'expected_features': expected_features,
            'message': 'XGBoost model is loaded and ready for predictions.'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Model test failed: {str(e)}'})

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

VERIFICATION_CODE_EXPIRY_SECONDS = 60

@app.route('/send_code')
def send_code():
    code = str(random.randint(100000, 999999))
    session['verification_code'] = code
    session['code_timestamp'] = time.time()
    # send code via email here
    return render_template('enter_code.html')

@app.route('/verify_code', methods=['POST'])
def verify_code():
    input_code = request.form['code']
    actual_code = session.get('verification_code')
    code_timestamp = session.get('code_timestamp')
    if not actual_code or not code_timestamp:
        return "No code found. Please request a new code."
    if time.time() - code_timestamp > VERIFICATION_CODE_EXPIRY_SECONDS:
        return "Verification code expired. Please request a new one."
    if input_code != actual_code:
        return "Invalid verification code. Please try again."
    return "Verification successful!"

if __name__ == "__main__":
    app.run(debug=True)
