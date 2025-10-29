from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

def train_model():
    """Train Random Forest model with improved parameters for imbalanced data"""
    global trained_model, scaler, feature_columns, feature_importance_dict
    
    try:
        print("Training Random Forest model with balanced settings...")
        
        # Fetch all data from MongoDB
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        
        if df.empty:
            print("Error: No data found in MongoDB")
            return False
        
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
        
        # Check class distribution
        fraud_count = y.sum()
        legit_count = len(y) - fraud_count
        print(f"Dataset: {legit_count} legitimate, {fraud_count} fraud transactions")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest with optimized parameters for fraud detection
        trained_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced_subsample',  # Better for highly imbalanced data
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        trained_model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        feature_importance_dict = dict(zip(feature_columns, trained_model.feature_importances_))
        
        # Evaluate model
        y_pred = trained_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"✓ Model trained successfully!")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
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
    """Get dataset statistics - cached in memory"""
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
    """Get all analytics data in one call - confusion matrix, DGIM, Bloom Filter, Girvan-Newman"""
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
        
        # DGIM Algorithm - Monitor risky transactions in sliding window
        dgim_result = apply_dgim_algorithm(df)
        
        # Bloom Filter - Check fraudulent identifiers
        bloom_result = apply_bloom_filter(df)
        
        # Girvan-Newman - Detect fraud communities
        girvan_newman_result = apply_girvan_newman(df)
        
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
            'bloom_filter': bloom_result,
            'girvan_newman': girvan_newman_result
        })
    except Exception as e:
        print(f"Error in /api/analytics-data: {e}")
        return jsonify({'error': str(e)}), 500

def apply_dgim_algorithm(df):
    """Apply DGIM algorithm for counting risky transactions in sliding window"""
    try:
        # Define risky transaction criteria
        # Example: Transaction amount > 75th percentile is considered risky
        if 'Amount' in df.columns:
            threshold = df['Amount'].quantile(0.75)
            risky_bits = (df['Amount'] > threshold).astype(int).tolist()
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                threshold = df[col].quantile(0.75)
                risky_bits = (df[col] > threshold).astype(int).tolist()
            else:
                risky_bits = [0] * len(df)
        
        # Apply DGIM
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
            'description': 'DGIM estimates risky transactions in a sliding window with minimal memory'
        }
    except Exception as e:
        print(f"Error in DGIM: {e}")
        return {'error': str(e)}

def apply_bloom_filter(df):
    """Apply Bloom Filter to detect known fraudulent patterns"""
    try:
        # Create Bloom Filter with fraud cases
        bloom = BloomFilter(size=10000, hash_count=5)
        
        fraud_df = df[df['Is_Fraud'] == 1]
        
        # Add fraudulent transaction identifiers to bloom filter
        fraud_identifiers = []
        for idx, row in fraud_df.iterrows():
            # Create unique identifier from transaction features
            identifier = f"{row.get('Amount', 0)}_{row.get('Is_Fraud', 0)}"
            bloom.add(identifier)
            fraud_identifiers.append(identifier)
        
        # Test bloom filter on all transactions
        test_results = {'true_positives': 0, 'false_positives': 0, 
                       'true_negatives': 0, 'false_negatives': 0}
        
        for idx, row in df.iterrows():
            identifier = f"{row.get('Amount', 0)}_{row.get('Is_Fraud', 0)}"
            in_filter = bloom.check(identifier)
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
            'description': 'Bloom Filter quickly identifies potentially fraudulent transaction patterns'
        }
    except Exception as e:
        print(f"Error in Bloom Filter: {e}")
        return {'error': str(e)}

def apply_girvan_newman(df):
    """Apply Girvan-Newman algorithm to detect fraud communities"""
    try:
        # Build transaction graph
        G = nx.Graph()
        
        # Sample data if too large
        sample_size = min(500, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # Add nodes (transactions)
        for idx, row in df_sample.iterrows():
            G.add_node(idx, fraud=row['Is_Fraud'])
        
        # Add edges between similar transactions
        # Connect transactions with similar amounts
        if 'Amount' in df_sample.columns:
            amounts = df_sample['Amount'].values
            indices = df_sample.index.values
            
            for i in range(len(amounts)):
                for j in range(i + 1, min(i + 20, len(amounts))):  # Limit connections
                    if abs(amounts[i] - amounts[j]) < amounts[i] * 0.1:  # Within 10% similarity
                        G.add_edge(indices[i], indices[j])
        
        if G.number_of_edges() == 0:
            return {
                'communities_found': 0,
                'description': 'No connected transactions found for community detection'
            }
        
        # Apply Girvan-Newman (limited iterations for performance)
        communities = []
        k = min(5, G.number_of_nodes() // 10)  # Find ~5 communities
        
        comp = nx.community.girvan_newman(G)
        for communities in range(k):
            try:
                communities = next(comp)
            except StopIteration:
                break
        
        # Analyze communities
        if communities:
            community_analysis = []
            for i, community in enumerate(communities):
                community_nodes = list(community)
                fraud_count = sum(1 for node in community_nodes if df_sample.loc[node, 'Is_Fraud'] == 1)
                fraud_rate = round((fraud_count / len(community_nodes)) * 100, 2) if community_nodes else 0
                
                community_analysis.append({
                    'community_id': i + 1,
                    'size': len(community_nodes),
                    'fraud_count': fraud_count,
                    'fraud_rate': fraud_rate,
                    'risk_level': 'High' if fraud_rate > 50 else ('Medium' if fraud_rate > 20 else 'Low')
                })
            
            return {
                'communities_found': len(communities),
                'community_analysis': community_analysis,
                'description': 'Girvan-Newman detects groups of related transactions to identify fraud networks'
            }
        
        return {
            'communities_found': 0,
            'description': 'No distinct communities detected'
        }
        
    except Exception as e:
        print(f"Error in Girvan-Newman: {e}")
        return {'error': str(e)}

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
        
        with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
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
            print("✓ Random Forest model ready for predictions")
        else:
            print("✗ Failed to train model")
            
    except Exception as e:
        print(f"✗ MongoDB connection error: {e}")
    
    app.run(debug=True, port=5000)