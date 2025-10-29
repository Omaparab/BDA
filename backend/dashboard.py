from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure Flask to use frontend folder for templates
app = Flask(__name__, template_folder='../frontend')

# MongoDB Connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['transactions']
collection = db['bda']

# Global variables for trained model
trained_model = None
scaler = None
feature_columns = None
feature_importance_dict = None

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

def train_model():
    """Train Random Forest model on the dataset"""
    global trained_model, scaler, feature_columns, feature_importance_dict
    
    try:
        print("Training Random Forest model...")
        
        # Fetch all data from MongoDB
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        # Remove non-numeric columns and _id
        df = df.drop(columns=['_id'], errors='ignore')
        
        # Identify target column
        if 'Is_Fraud' not in df.columns:
            print("Error: 'Is_Fraud' column not found")
            return False
        
        # Separate features and target
        X = df.drop(columns=['Is_Fraud'])
        y = df['Is_Fraud']
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        feature_columns = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest with class balancing
        trained_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Critical for imbalanced data
            random_state=42,
            n_jobs=-1
        )
        trained_model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        feature_importance_dict = dict(zip(feature_columns, trained_model.feature_importances_))
        
        # Evaluate model
        y_pred = trained_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Model trained successfully! Accuracy: {accuracy:.2%}")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def flajolet_martin(values):
    """
    Flajolet-Martin algorithm for estimating unique count
    Uses multiple hash functions for better accuracy
    """
    def hash_value(val, seed):
        """Hash function with seed"""
        hash_obj = hashlib.md5(f"{val}{seed}".encode())
        return int(hash_obj.hexdigest(), 16)
    
    def trailing_zeros(n):
        """Count trailing zeros in binary representation"""
        if n == 0:
            return 0
        count = 0
        while (n & 1) == 0:
            count += 1
            n >>= 1
        return count
    
    # Use multiple hash functions for better accuracy
    num_hash_functions = 10
    max_trailing_zeros = [0] * num_hash_functions
    
    for value in values:
        for i in range(num_hash_functions):
            hashed = hash_value(value, i)
            zeros = trailing_zeros(hashed)
            max_trailing_zeros[i] = max(max_trailing_zeros[i], zeros)
    
    # Calculate estimates from each hash function
    estimates = [2 ** r for r in max_trailing_zeros]
    
    # Return median estimate (more robust than mean)
    return int(np.median(estimates))

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

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
    """Get dataset statistics from MongoDB"""
    try:
        cursor = collection.find({}, {'Is_Fraud': 1})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        if df.empty or 'Is_Fraud' not in df.columns:
            return jsonify({
                'total_transactions': 0,
                'fraud_cases': 0,
                'legitimate_cases': 0,
                'fraud_percentage': 0.0
            })
        
        total_transactions = len(df)
        fraud_cases = int(df['Is_Fraud'].sum())
        legitimate_cases = int((df['Is_Fraud'] == 0).sum())
        fraud_percentage = round(df['Is_Fraud'].mean() * 100, 2)
        
        return jsonify({
            'total_transactions': total_transactions,
            'fraud_cases': fraud_cases,
            'legitimate_cases': legitimate_cases,
            'fraud_percentage': fraud_percentage
        })
    except Exception as e:
        print(f"Error in /api/stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/confusion-matrix')
def get_confusion_matrix():
    """Get confusion matrix and model metrics"""
    try:
        if trained_model is None or scaler is None:
            return jsonify({'error': 'Model not trained'}), 500
        
        # Fetch all data
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        # Prepare data
        df = df.drop(columns=['_id'], errors='ignore')
        X = df[feature_columns]
        y = df['Is_Fraud']
        
        X = X.fillna(X.mean())
        
        # Split and scale
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_scaled = scaler.transform(X_test)
        
        # Predict
        y_pred = trained_model.predict(X_test_scaled)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate metrics
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        precision = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
        recall = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
        
        return jsonify({
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            },
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'feature_importance': feature_importance_dict
        })
    except Exception as e:
        print(f"Error in /api/confusion-matrix: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/flajolet-martin')
def get_flajolet_martin():
    """Apply Flajolet-Martin algorithm to estimate unique values"""
    try:
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        df = df.drop(columns=['_id'], errors='ignore')
        
        unique_counts = {}
        
        # Apply FM algorithm to each column
        for column in df.columns:
            if column != 'Is_Fraud':
                values = df[column].astype(str).tolist()
                estimated_unique = flajolet_martin(values)
                unique_counts[column] = estimated_unique
        
        return jsonify({
            'unique_counts': unique_counts,
            'algorithm': 'Flajolet-Martin',
            'description': 'Probabilistic algorithm for cardinality estimation'
        })
    except Exception as e:
        print(f"Error in /api/flajolet-martin: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-features')
def get_model_features():
    """Get list of features required for prediction"""
    try:
        if feature_columns is None:
            return jsonify({'error': 'Model not trained'}), 500
        
        return jsonify({
            'features': feature_columns
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
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([data], columns=feature_columns)
        
        # Fill missing values with 0
        input_df = input_df.fillna(0)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Connecting to MongoDB: {MONGO_URI}")
    print(f"Database: transactions, Collection: bda")
    
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
            print("✓ Random Forest model ready for predictions")
        else:
            print("✗ Failed to train model")
            
    except Exception as e:
        print(f"✗ MongoDB connection error: {e}")
    
    app.run(debug=True, port=5000)