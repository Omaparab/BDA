from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import os
import json

# Configure Flask to use frontend folder for templates
app = Flask(__name__, template_folder='../frontend')

# MongoDB Connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['transactions']
collection = db['bda']

def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable format"""
    if doc is None:
        return None
    
    serialized = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, bytes):
            # Convert binary data to string representation
            serialized[key] = str(value)
        elif hasattr(value, '__dict__'):
            # Handle any other complex objects
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """Get paginated transaction data with filtering"""
    try:
        # Get query parameters
        fraud_filter = request.args.get('filter', 'all')
        page = int(request.args.get('page', 1))
        per_page = 20
        
        # Build MongoDB query
        query = {}
        if fraud_filter == 'fraud':
            query = {'Is_Fraud': 1}
        elif fraud_filter == 'legit':
            query = {'Is_Fraud': 0}
        
        # Get total counts
        total_records = collection.count_documents({})
        filtered_records = collection.count_documents(query)
        
        # Pagination
        skip = (page - 1) * per_page
        
        # Fetch data from MongoDB
        cursor = collection.find(query).skip(skip).limit(per_page)
        data_raw = list(cursor)
        
        # Serialize all documents
        data_sample = [serialize_doc(doc) for doc in data_raw]
        
        # Get total number of fields from first document (excluding _id)
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
    """Get dataset statistics from MongoDB using aggregation"""
    try:
        # Fetch all data from MongoDB
        cursor = collection.find({}, {'Is_Fraud': 1})
        data = list(cursor)
        
        # Convert to pandas DataFrame for easier calculation
        df = pd.DataFrame(data)
        
        if df.empty or 'Is_Fraud' not in df.columns:
            return jsonify({
                'total_transactions': 0,
                'fraud_cases': 0,
                'legitimate_cases': 0,
                'fraud_percentage': 0.0
            })
        
        # Calculate statistics
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
        
        # Show sample document structure
        sample = collection.find_one()
        if sample:
            print(f"✓ Sample document fields: {list(sample.keys())}")
    except Exception as e:
        print(f"✗ MongoDB connection error: {e}")
    
    app.run(debug=True, port=5000)