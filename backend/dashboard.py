from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# Configure Flask to use frontend folder for templates
app = Flask(__name__, template_folder='../frontend')

# Global variables for caching
model = None
metrics = None
data_cache = None

def load_and_preprocess():
    """Load and preprocess data with minimal code"""
    train = pd.read_csv('../data/training_data.csv')
    test = pd.read_csv('../data/test_data.csv')
    
    # Select relevant features
    feature_cols = ['Age', 'Transaction_Amount', 'Account_Type', 'Merchant_Type', 
                    'Transaction_Type', 'Device_Type']
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in ['Account_Type', 'Merchant_Type', 'Transaction_Type', 'Device_Type']:
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    X_train = train[feature_cols]
    y_train = train['Is_Fraud']
    X_test = test[feature_cols]
    y_test = test['Is_Fraud']
    
    return X_train, y_train, X_test, y_test, train, test

def train_model():
    """Train Random Forest with parallel processing"""
    global model, metrics
    
    X_train, y_train, X_test, y_test, _, _ = load_and_preprocess()
    
    # Random Forest with parallel processing (n_jobs=-1 uses all CPU cores)
    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,  # KEY: Use all available cores for fast training
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall': round(recall_score(y_test, y_pred) * 100, 2),
        'f1_score': round(f1_score(y_test, y_pred) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """Get paginated transaction data with filtering"""
    global data_cache
    if data_cache is None:
        data_cache = pd.read_csv('../data/Bank_Transaction_Fraud_Detection.csv')
    
    # Get query parameters
    fraud_filter = request.args.get('filter', 'all')
    page = int(request.args.get('page', 1))
    per_page = 20
    
    # Filter data
    if fraud_filter == 'fraud':
        filtered_data = data_cache[data_cache['Is_Fraud'] == 1]
    elif fraud_filter == 'legit':
        filtered_data = data_cache[data_cache['Is_Fraud'] == 0]
    else:
        filtered_data = data_cache
    
    # Select top 10 important features
    important_cols = ['Customer_Name', 'Age', 'Transaction_Amount', 'Account_Type', 
                     'Merchant_Category', 'Transaction_Type', 'Device_Type', 
                     'Transaction_Location', 'Account_Balance', 'Is_Fraud']
    
    # Pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    data_sample = filtered_data[important_cols].iloc[start_idx:end_idx].to_dict('records')
    
    return jsonify({
        'data': data_sample,
        'total_records': len(data_cache),
        'filtered_records': len(filtered_data),
        'total_features': len(data_cache.columns),
        'page': page,
        'per_page': per_page,
        'total_pages': (len(filtered_data) + per_page - 1) // per_page
    })

@app.route('/api/metrics')
def get_metrics():
    """Get model performance metrics"""
    global metrics
    if metrics is None:
        train_model()
    return jsonify(metrics)

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    global data_cache
    if data_cache is None:
        data_cache = pd.read_csv('../data/Bank_Transaction_Fraud_Detection.csv')
    
    return jsonify({
        'total_transactions': len(data_cache),
        'fraud_cases': int(data_cache['Is_Fraud'].sum()),
        'legitimate_cases': int((data_cache['Is_Fraud'] == 0).sum()),
        'fraud_percentage': round(data_cache['Is_Fraud'].mean() * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)