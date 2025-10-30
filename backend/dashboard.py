from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import defaultdict, deque
import os
import hashlib
import warnings
import csv
from datetime import datetime
import networkx as nx
warnings.filterwarnings('ignore')

# Configure Flask to use frontend folder for templates
app = Flask(__name__, template_folder='../frontend')

# MongoDB Connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['transactions']
collection = db['bda']

# CSV File Path
CSV_FILE_PATH = 'transactions_data.csv'

# Global variables for trained model
trained_model = None
scaler = None
feature_columns = None
feature_importance_dict = None
label_encoders = {}  # Store label encoders for categorical variables
sampling_info = {}  # Store information about data balancing

# Define all expected fields for the transaction form
TRANSACTION_FIELDS = [
    'Customer_Name', 'Gender', 'Age', 'State', 'City', 'Bank_Branch',
    'Account_Type', 'Transaction_Date', 'Transaction_Time', 'Transaction_Amount',
    'Transaction_Type', 'Merchant_Category', 'Account_Balance', 'Transaction_Device',
    'Transaction_Location', 'Device_Type', 'Transaction_Currency', 'Customer_Contact',
    'Transaction_Description', 'Is_Fraud'
]

def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable format"""
    if doc is None:
        return None
    
    serialized = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, bytes):
            serialized[key] = str(value)
        elif hasattr(value, '__dict__'):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized

def engineer_features(df):
    """Convert raw transaction data into features suitable for ML model"""
    global label_encoders
    
    # Create a copy to avoid modifying original
    df_features = df.copy()
    
    # Numeric features that can be used directly
    numeric_features = ['Age', 'Transaction_Amount', 'Account_Balance']
    
    # Categorical features that need encoding
    categorical_features = ['Gender', 'State', 'Account_Type', 'Transaction_Type', 
                           'Merchant_Category', 'Transaction_Device', 'Device_Type', 
                           'Transaction_Currency']
    
    # Encode categorical variables
    for col in categorical_features:
        if col in df_features.columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                # Fit on all possible values
                df_features[col] = df_features[col].fillna('Unknown')
                label_encoders[col].fit(df_features[col])
            else:
                df_features[col] = df_features[col].fillna('Unknown')
            
            # Transform
            try:
                df_features[col + '_Encoded'] = label_encoders[col].transform(df_features[col])
            except ValueError:
                # Handle unseen categories
                df_features[col + '_Encoded'] = 0
            
            # Drop original categorical column
            df_features = df_features.drop(columns=[col])
    
    # Parse Transaction_Date and Transaction_Time to extract features
    if 'Transaction_Date' in df_features.columns:
        try:
            # Handle different date formats
            df_features['Transaction_Date'] = pd.to_datetime(df_features['Transaction_Date'], 
                                                             format='%d-%m-%Y', errors='coerce')
            df_features['Day_of_Week'] = df_features['Transaction_Date'].dt.dayofweek
            df_features['Day_of_Month'] = df_features['Transaction_Date'].dt.day
            df_features['Month'] = df_features['Transaction_Date'].dt.month
            df_features = df_features.drop(columns=['Transaction_Date'])
        except:
            df_features = df_features.drop(columns=['Transaction_Date'], errors='ignore')
    
    if 'Transaction_Time' in df_features.columns:
        try:
            # Extract hour from time
            df_features['Transaction_Time'] = pd.to_datetime(df_features['Transaction_Time'], 
                                                             format='%H:%M:%S', errors='coerce')
            df_features['Hour'] = df_features['Transaction_Time'].dt.hour
            df_features = df_features.drop(columns=['Transaction_Time'])
        except:
            df_features = df_features.drop(columns=['Transaction_Time'], errors='ignore')
    
    # Drop text fields that can't be easily encoded
    text_fields = ['Customer_Name', 'City', 'Bank_Branch', 'Transaction_Location', 
                   'Customer_Contact', 'Transaction_Description', 'timestamp']
    df_features = df_features.drop(columns=text_fields, errors='ignore')
    
    # Keep only numeric columns
    df_features = df_features.select_dtypes(include=[np.number])
    
    return df_features

def train_model():
    """Train Random Forest model with SMOTE for imbalanced data"""
    global trained_model, scaler, feature_columns, feature_importance_dict, sampling_info
    
    try:
        print("Training Random Forest model with imbalanced-learn SMOTE...")
        
        # Fetch all data from MongoDB
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        if df.empty:
            print("Error: No data found in MongoDB")
            return False
        
        # Remove _id
        df = df.drop(columns=['_id'], errors='ignore')
        
        # Identify target column
        if 'Is_Fraud' not in df.columns:
            print("Error: 'Is_Fraud' column not found")
            return False
        
        # Separate target
        y = df['Is_Fraud']
        
        # Engineer features
        X = engineer_features(df.drop(columns=['Is_Fraud']))
        
        if X.empty or len(X.columns) == 0:
            print("Error: No features available after engineering")
            return False
        
        feature_columns = X.columns.tolist()
        print(f"Features used for training: {feature_columns}")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Check ORIGINAL class distribution
        fraud_count = y.sum()
        legit_count = len(y) - fraud_count
        imbalance_ratio = legit_count / max(fraud_count, 1)
        
        print(f"\n📊 ORIGINAL Dataset Distribution:")
        print(f"   Legitimate: {legit_count} ({(legit_count/len(y)*100):.2f}%)")
        print(f"   Fraud: {fraud_count} ({(fraud_count/len(y)*100):.2f}%)")
        print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if fraud_count == 0 or legit_count == 0:
            print("Warning: Imbalanced dataset with only one class")
            return False
        
        # Split data BEFORE resampling to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store original training distribution
        original_train_fraud = y_train.sum()
        original_train_legit = len(y_train) - original_train_fraud
        
        # Apply SMOTE only to training data
        print(f"\n🔄 Applying SMOTE to balance training data...")
        
        # Choose resampling strategy based on imbalance severity
        if imbalance_ratio > 20:  # Severe imbalance
            print("   Using SMOTETomek (SMOTE + Tomek Links) for severe imbalance")
            resampler = SMOTETomek(
                sampling_strategy='auto',  # Balance to 1:1 ratio
                random_state=42
            )
        elif imbalance_ratio > 5:  # Moderate imbalance
            print("   Using SMOTE with auto sampling strategy")
            resampler = SMOTE(
                sampling_strategy='auto',  # Balance to 1:1 ratio
                random_state=42,
                k_neighbors=min(5, fraud_count - 1)  # Adjust based on fraud samples
            )
        else:  # Mild imbalance
            print("   Using SMOTE with 0.5 sampling strategy (moderate balancing)")
            resampler = SMOTE(
                sampling_strategy=0.5,  # Fraud will be 50% of legitimate
                random_state=42,
                k_neighbors=5
            )
        
        # Resample training data
        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        
        # Calculate NEW distribution
        resampled_fraud = y_train_resampled.sum()
        resampled_legit = len(y_train_resampled) - resampled_fraud
        
        print(f"\n📊 RESAMPLED Training Data Distribution:")
        print(f"   Before SMOTE - Legitimate: {original_train_legit}, Fraud: {original_train_fraud}")
        print(f"   After SMOTE  - Legitimate: {resampled_legit}, Fraud: {resampled_fraud}")
        print(f"   New Ratio: {resampled_legit/resampled_fraud:.2f}:1")
        print(f"   Synthetic samples created: {resampled_fraud - original_train_fraud}")
        
        # Store sampling information
        sampling_info = {
            'original_train_legit': int(original_train_legit),
            'original_train_fraud': int(original_train_fraud),
            'resampled_train_legit': int(resampled_legit),
            'resampled_train_fraud': int(resampled_fraud),
            'synthetic_samples': int(resampled_fraud - original_train_fraud),
            'original_ratio': round(imbalance_ratio, 2),
            'resampled_ratio': round(resampled_legit/resampled_fraud, 2)
        }
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest with adjusted parameters for balanced data
        print(f"\n🌲 Training Random Forest...")
        trained_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight=None,  # CHANGED: No need for class_weight with SMOTE
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        trained_model.fit(X_train_scaled, y_train_resampled)
        
        # Calculate feature importance
        feature_importance_dict = dict(zip(feature_columns, trained_model.feature_importances_))
        
        # Evaluate model on ORIGINAL (unbalanced) test set
        print(f"\n📈 Model Evaluation (on original test set):")
        y_pred = trained_model.predict(X_test_scaled)
        y_pred_proba = trained_model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"   ✓ Accuracy:  {accuracy:.2%}")
        print(f"   ✓ Precision: {precision:.2%} (of flagged frauds, how many are real)")
        print(f"   ✓ Recall:    {recall:.2%} (of real frauds, how many we catch)")
        print(f"   ✓ F1 Score:  {f1:.2%} (harmonic mean)")
        print(f"\n   Confusion Matrix:")
        print(f"   True Negatives:  {tn} (correctly identified legit)")
        print(f"   False Positives: {fp} (legit flagged as fraud)")
        print(f"   False Negatives: {fn} (fraud missed)")
        print(f"   True Positives:  {tp} (correctly identified fraud)")
        
        # Calculate fraud detection rate
        fraud_detection_rate = (tp / max(tp + fn, 1)) * 100
        false_alarm_rate = (fp / max(fp + tn, 1)) * 100
        
        print(f"\n   🎯 Fraud Detection Rate: {fraud_detection_rate:.2f}%")
        print(f"   ⚠️  False Alarm Rate: {false_alarm_rate:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Routes
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/add-transaction')
def add_transaction():
    return render_template('add_transaction.html')

@app.route('/api/data')
def get_data():
    """Get paginated transaction data with filtering"""
    try:
        fraud_filter = request.args.get('filter', 'all')
        page = int(request.args.get('page', 1))
        per_page = 20
        
        query = {}
        if fraud_filter == 'fraud':
            query = {'Is_Fraud': 1}
        elif fraud_filter == 'legit':
            query = {'Is_Fraud': 0}
        
        total_records = collection.count_documents({})
        filtered_records = collection.count_documents(query)
        
        skip = (page - 1) * per_page
        cursor = collection.find(query).skip(skip).limit(per_page)
        data_raw = list(cursor)
        
        data_sample = [serialize_doc(doc) for doc in data_raw]
        
        total_features = len(data_sample[0].keys()) - 1 if data_sample and '_id' in data_sample[0] else (len(data_sample[0].keys()) if data_sample else 0)
        
        return jsonify({
            'data': data_sample,
            'total_records': total_records,
            'filtered_records': filtered_records,
            'total_features': total_features,
            'page': page,
            'per_page': per_page,
            'total_pages': (filtered_records + per_page - 1) // per_page if filtered_records > 0 else 1
        })
    except Exception as e:
        print(f"Error in /api/data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics including SMOTE information"""
    try:
        cursor = collection.find({}, {'Is_Fraud': 1})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        if df.empty or 'Is_Fraud' not in df.columns:
            return jsonify({
                'total_transactions': 0,
                'fraud_cases': 0,
                'legitimate_cases': 0,
                'fraud_percentage': 0.0,
                'sampling_info': {}
            })
        
        total_transactions = len(df)
        fraud_cases = int(df['Is_Fraud'].sum())
        legitimate_cases = int((df['Is_Fraud'] == 0).sum())
        fraud_percentage = round(df['Is_Fraud'].mean() * 100, 2)
        
        return jsonify({
            'total_transactions': total_transactions,
            'fraud_cases': fraud_cases,
            'legitimate_cases': legitimate_cases,
            'fraud_percentage': fraud_percentage,
            'sampling_info': sampling_info  # Include SMOTE statistics
        })
    except Exception as e:
        print(f"Error in /api/stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-features')
def get_model_features():
    """Get list of fields required for the transaction form"""
    try:
        return jsonify({
            'features': TRANSACTION_FIELDS
        })
    except Exception as e:
        print(f"Error in /api/model-features: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if a transaction is fraudulent"""
    try:
        if trained_model is None or scaler is None:
            return jsonify({'error': 'Model not trained'}), 500
        
        data = request.json
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Engineer features
        input_features = engineer_features(input_df)
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in input_features.columns:
                input_features[col] = 0
        
        input_features = input_features[feature_columns]  # Reorder columns
        input_features = input_features.fillna(0)
        
        # Scale features
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = int(trained_model.predict(input_scaled)[0])
        probability = round(float(trained_model.predict_proba(input_scaled)[0][prediction]) * 100, 2)
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'label': 'Fraudulent' if prediction == 1 else 'Legitimate'
        })
    except Exception as e:
        print(f"Error in /api/predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-transaction', methods=['POST'])
def add_transaction_api():
    """Add new transaction to MongoDB and CSV"""
    try:
        data = request.json
        
        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()
        
        # Insert into MongoDB
        result = collection.insert_one(data)
        
        # Append to CSV
        csv_exists = os.path.exists(CSV_FILE_PATH)
        
        with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            
            # Write header if file is new
            if not csv_exists or os.path.getsize(CSV_FILE_PATH) == 0:
                writer.writeheader()
            
            writer.writerow(data)
        
        # Retrain model with new data
        train_model()
        
        return jsonify({
            'success': True,
            'message': 'Transaction added successfully',
            'id': str(result.inserted_id)
        })
        
    except Exception as e:
        print(f"Error adding transaction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Connecting to MongoDB: {MONGO_URI}")
    print(f"Database: transactions, Collection: bda")
    print(f"CSV File: {CSV_FILE_PATH}")
    
    # Test MongoDB connection
    try:
        client.server_info()
        print("✓ MongoDB connected successfully")
        doc_count = collection.count_documents({})
        print(f"✓ Found {doc_count} documents in collection")
        
        sample = collection.find_one()
        if sample:
            print(f"✓ Sample document fields: {list(sample.keys())}")
        
        # Train the model
        if train_model():
            print("\n✓ Random Forest model ready for predictions with SMOTE balancing")
        else:
            print("\n✗ Failed to train model")
            
    except Exception as e:
        print(f"✗ MongoDB connection error: {e}")
    
    app.run(debug=True, port=5000)