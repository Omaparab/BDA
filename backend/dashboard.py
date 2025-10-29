from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import defaultdict
import os
import hashlib
import warnings
import csv
from datetime import datetime
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
label_encoders = {}

# Define all expected fields for the transaction form
TRANSACTION_FIELDS = [
    'Customer_Name', 'Gender', 'Age', 'State', 'City', 'Bank_Branch',
    'Account_Type', 'Transaction_Date', 'Transaction_Time', 'Transaction_Amount',
    'Transaction_Type', 'Merchant_Category', 'Account_Balance', 'Transaction_Device',
    'Transaction_Location', 'Device_Type', 'Transaction_Currency', 'Customer_Contact',
    'Transaction_Description', 'Is_Fraud'
]

# DGIM Algorithm Implementation
class DGIMBucket:
    def __init__(self, timestamp, size):
        self.timestamp = timestamp
        self.size = size

class DGIM:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.buckets = []
        self.current_timestamp = 0
    
    def update(self, bit):
        self.current_timestamp += 1
        
        if bit == 1:
            self.buckets.append(DGIMBucket(self.current_timestamp, 1))
            self._merge_buckets()
        
        # Remove old buckets outside window
        self.buckets = [b for b in self.buckets 
                       if self.current_timestamp - b.timestamp < self.window_size]
    
    def _merge_buckets(self):
        # Merge buckets of same size if there are more than 2
        size_count = defaultdict(int)
        for bucket in self.buckets:
            size_count[bucket.size] += 1
        
        for size in sorted(size_count.keys()):
            if size_count[size] > 2:
                # Merge oldest two buckets of this size
                buckets_of_size = [b for b in self.buckets if b.size == size]
                buckets_of_size.sort(key=lambda x: x.timestamp)
                
                if len(buckets_of_size) >= 2:
                    b1, b2 = buckets_of_size[0], buckets_of_size[1]
                    self.buckets.remove(b1)
                    self.buckets.remove(b2)
                    self.buckets.append(DGIMBucket(b2.timestamp, size * 2))
    
    def query(self):
        # Estimate count of 1s in window
        if not self.buckets:
            return 0
        
        total = sum(b.size for b in self.buckets[:-1])
        # Add half of the oldest bucket as estimate
        if self.buckets:
            total += self.buckets[-1].size // 2
        
        return total

# Bloom Filter Implementation
class BloomFilter:
    def __init__(self, size=10000, hash_count=5):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size
    
    def _hash(self, item, seed):
        hash_obj = hashlib.md5(f"{item}{seed}".encode())
        return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, item):
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
    
    def check(self, item):
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True

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
    """Train XGBoost model with SMOTE to handle class imbalance - Optimized for >80% performance"""
    global trained_model, scaler, feature_columns, feature_importance_dict
    
    try:
        print("\n" + "="*60)
        print("Training XGBoost model with SMOTE for class balance...")
        print("="*60)
        
        # Fetch all data from MongoDB
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        if df.empty:
            print("❌ Error: No data found in MongoDB")
            return False
        
        # Remove _id
        df = df.drop(columns=['_id'], errors='ignore')
        
        # Identify target column
        if 'Is_Fraud' not in df.columns:
            print("❌ Error: 'Is_Fraud' column not found")
            return False
        
        # Separate target
        y = df['Is_Fraud']
        
        # Engineer features
        X = engineer_features(df.drop(columns=['Is_Fraud']))
        
        if X.empty or len(X.columns) == 0:
            print("❌ Error: No features available after engineering")
            return False
        
        feature_columns = X.columns.tolist()
        print(f"\n📊 Features used for training ({len(feature_columns)}): {feature_columns[:5]}...")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Check class distribution BEFORE balancing
        fraud_count = y.sum()
        legit_count = len(y) - fraud_count
        print(f"\n⚖️  Original Dataset Distribution:")
        print(f"   Legitimate: {legit_count:,} ({legit_count/len(y)*100:.2f}%)")
        print(f"   Fraud: {fraud_count:,} ({fraud_count/len(y)*100:.2f}%)")
        print(f"   Imbalance Ratio: 1:{legit_count/max(fraud_count, 1):.1f}")
        
        if fraud_count == 0 or legit_count == 0:
            print("⚠️  Warning: Imbalanced dataset with only one class")
            return False
        
        # Split data BEFORE SMOTE (important!)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Train set: {len(X_train):,} samples")
        print(f"📉 Test set: {len(X_test):,} samples")
        
        # Apply SMOTE ONLY to training data
        print("\n🔄 Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, fraud_count-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        fraud_train_balanced = y_train_balanced.sum()
        legit_train_balanced = len(y_train_balanced) - fraud_train_balanced
        print(f"✅ Balanced Training Set:")
        print(f"   Legitimate: {legit_train_balanced:,}")
        print(f"   Fraud: {fraud_train_balanced:,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = legit_count / max(fraud_count, 1)
        
        # Train XGBoost with optimized hyperparameters for >80% performance
        print("\n🚀 Training XGBoost Classifier with optimized parameters...")
        trained_model = xgb.XGBClassifier(
            # Core parameters
            n_estimators=300,              # More trees for better learning
            max_depth=8,                   # Deeper trees for complex patterns
            learning_rate=0.05,            # Lower learning rate for better generalization
            
            # Regularization (prevent overfitting)
            min_child_weight=3,            # Minimum sum of instance weight in a child
            gamma=0.1,                     # Minimum loss reduction for split
            subsample=0.8,                 # Subsample ratio of training instances
            colsample_bytree=0.8,          # Subsample ratio of columns per tree
            reg_alpha=0.1,                 # L1 regularization
            reg_lambda=1.0,                # L2 regularization
            
            # Class imbalance handling
            scale_pos_weight=scale_pos_weight,  # Balance positive/negative weights
            
            # Performance optimization
            tree_method='hist',            # Faster histogram-based algorithm
            objective='binary:logistic',   # Binary classification
            eval_metric='auc',             # Area under ROC curve
            
            # Reproducibility
            random_state=42,
            n_jobs=-1,                     # Use all CPU cores
            verbosity=0                    # Suppress warnings
        )
        
        # Fit with evaluation set for early stopping
        trained_model.fit(
            X_train_scaled, y_train_balanced,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Calculate feature importance
        feature_importance_dict = dict(zip(feature_columns, trained_model.feature_importances_))
        
        # Evaluate model
        print("\n🎯 Model Evaluation on Test Set:")
        y_pred = trained_model.predict(X_test_scaled)
        y_pred_proba = trained_model.predict_proba(X_test_scaled)[:, 1]
        
        # Detailed metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Additional metric for binary classification
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"   ROC-AUC:   {roc_auc:.2%} - Model's ability to distinguish classes")
        except:
            roc_auc = 0.0
        
        # Confusion matrix breakdown
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"\n   Confusion Matrix:")
        print(f"   True Positives (TP):  {tp:,} - Correctly identified frauds")
        print(f"   True Negatives (TN):  {tn:,} - Correctly identified legitimate")
        print(f"   False Positives (FP): {fp:,} - Legitimate flagged as fraud")
        print(f"   False Negatives (FN): {fn:,} - Fraud missed")
        
        print(f"\n   📊 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.2%} - Overall correctness")
        print(f"   Precision: {precision:.2%} - Of predicted frauds, how many were correct")
        print(f"   Recall:    {recall:.2%} - Of actual frauds, how many were caught")
        print(f"   F1 Score:  {f1:.2%} - Balanced metric")
        
        # Performance check
        if accuracy >= 0.80 and f1 >= 0.80:
            print(f"\n   ✅ EXCELLENT: Model achieves >80% accuracy and F1 score!")
        elif accuracy >= 0.80 or f1 >= 0.80:
            print(f"\n   ✅ GOOD: Model achieves >80% on at least one key metric")
        else:
            print(f"\n   ⚠️  WARNING: Model performance below 80% threshold")
            print(f"   Consider: More data, feature engineering, or hyperparameter tuning")
        
        # Display top 5 most important features
        if feature_importance_dict:
            print(f"\n   🔑 Top 5 Most Important Features:")
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            for idx, (feat, importance) in enumerate(sorted_features, 1):
                print(f"   {idx}. {feat}: {importance:.4f}")
        
        print("\n" + "="*60)
        print("✅ XGBoost model trained successfully!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
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
    """Get dataset statistics"""
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

@app.route('/api/analytics-data')
def get_analytics_data():
    """Get all analytics data in one call"""
    try:
        if trained_model is None or scaler is None:
            return jsonify({'error': 'Model not trained'}), 500
        
        # Fetch all data
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        # Prepare data
        df = df.drop(columns=['_id'], errors='ignore')
        y = df['Is_Fraud']
        X = engineer_features(df.drop(columns=['Is_Fraud']))
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_columns]  # Reorder columns
        X = X.fillna(X.mean())
        
        # Split and scale (same as training)
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
        
        # DGIM Algorithm
        dgim_result = apply_dgim_algorithm(df)
        
        # Bloom Filter
        bloom_result = apply_bloom_filter(df)
        
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
            'feature_importance': feature_importance_dict,
            'dgim': dgim_result,
            'bloom_filter': bloom_result
        })
    except Exception as e:
        print(f"Error in /api/analytics-data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def apply_dgim_algorithm(df):
    """
    DGIM Algorithm: Estimates count of 1s in a sliding window using logarithmic memory.
    
    How it's used in this project:
    1. Monitors real-time stream of transactions
    2. Identifies "risky" transactions (amount > 75th percentile)
    3. Maintains sliding window of last 1000 transactions
    4. Uses exponentially-sized buckets to estimate count with minimal memory
    5. Provides approximate count of risky transactions in current window
    
    Benefit: O(log²N) memory instead of O(N) for exact counting
    """
    try:
        if 'Transaction_Amount' in df.columns:
            threshold = df['Transaction_Amount'].quantile(0.75)
            risky_bits = (df['Transaction_Amount'] > threshold).astype(int).tolist()
        else:
            risky_bits = [0] * len(df)
        
        dgim = DGIM(window_size=min(1000, len(risky_bits)))
        
        for bit in risky_bits:
            dgim.update(bit)
        
        estimated_risky = dgim.query()
        actual_risky = sum(risky_bits[-dgim.window_size:]) if len(risky_bits) >= dgim.window_size else sum(risky_bits)
        
        return {
            'window_size': dgim.window_size,
            'estimated_risky_count': estimated_risky,
            'actual_risky_count': actual_risky,
            'accuracy': round((1 - abs(estimated_risky - actual_risky) / max(actual_risky, 1)) * 100, 2),
            'description': 'DGIM estimates risky transactions (>75th percentile amount) in sliding window with O(log²N) memory'
        }
    except Exception as e:
        print(f"Error in DGIM: {e}")
        return {'error': str(e)}

def apply_bloom_filter(df):
    """
    Bloom Filter: Space-efficient probabilistic data structure for set membership testing.
    
    How it's used in this project:
    1. Stores "signatures" of known fraudulent transactions
    2. Signature = hash of (Customer_Name + Transaction_Amount + Merchant_Category)
    3. Uses 5 hash functions to set bits in 100,000-bit array
    4. Quick first-pass check: "Have we seen this fraud pattern before?"
    5. False positives possible, but NO false negatives
    
    Benefit: O(1) lookup time, uses 12.5KB memory vs potentially MBs for exact storage
    """
    try:
        # Larger filter size for better accuracy
        bloom = BloomFilter(size=100000, hash_count=5)
        
        fraud_df = df[df['Is_Fraud'] == 1]
        
        # Create more unique fraud signatures
        fraud_identifiers = []
        for idx, row in fraud_df.iterrows():
            # Create signature from multiple fields
            signature = f"{row.get('Customer_Name', 'Unknown')}_{row.get('Transaction_Amount', 0)}_{row.get('Merchant_Category', 'Unknown')}"
            bloom.add(signature)
            fraud_identifiers.append(signature)
        
        test_results = {'true_positives': 0, 'false_positives': 0, 
                       'true_negatives': 0, 'false_negatives': 0}
        
        # Test on all transactions
        for idx, row in df.iterrows():
            signature = f"{row.get('Customer_Name', 'Unknown')}_{row.get('Transaction_Amount', 0)}_{row.get('Merchant_Category', 'Unknown')}"
            in_filter = bloom.check(signature)
            is_fraud = row['Is_Fraud'] == 1
            
            if in_filter and is_fraud:
                test_results['true_positives'] += 1
            elif in_filter and not is_fraud:
                test_results['false_positives'] += 1
            elif not in_filter and not is_fraud:
                test_results['true_negatives'] += 1
            else:
                test_results['false_negatives'] += 1
        
        total = sum(test_results.values())
        
        return {
            'fraud_patterns_stored': len(fraud_identifiers),
            'filter_size': bloom.size,
            'hash_functions': bloom.hash_count,
            'test_results': test_results,
            'false_positive_rate': round((test_results['false_positives'] / max(total, 1)) * 100, 2),
            'description': 'Bloom Filter stores fraud signatures (Name+Amount+Category) for O(1) blacklist lookup'
        }
    except Exception as e:
        print(f"Error in Bloom Filter: {e}")
        return {'error': str(e)}

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
    """Predict if a transaction is fraudulent using XGBoost"""
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
    print("\n" + "="*60)
    print("🚀 Starting Fraud Detection System with XGBoost")
    print("="*60)
    print(f"MongoDB URI: {MONGO_URI}")
    print(f"Database: transactions")
    print(f"Collection: bda")
    print(f"CSV File: {CSV_FILE_PATH}")
    print(f"Algorithm: XGBoost Classifier")
    print("="*60)
    
    # Test MongoDB connection
    try:
        client.server_info()
        print("✅ MongoDB connected successfully")
        doc_count = collection.count_documents({})
        print(f"✅ Found {doc_count:,} documents in collection")
        
        sample = collection.find_one()
        if sample:
            print(f"✅ Sample document fields: {list(sample.keys())[:5]}...")
        
        # Train the model
        if train_model():
            print("\n✅ System ready for fraud detection with XGBoost!")
        else:
            print("\n❌ Failed to train model - check data quality")
            
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
    
    print("\n" + "="*60)
    print("🌐 Starting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)