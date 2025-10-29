import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create data folder if it doesn't exist
os.makedirs('..\\data', exist_ok=True)

# Define file paths
train_path = os.path.join('..\\data', 'training_data.csv')
test_path = os.path.join('..\\data', 'test_data.csv')

# Check if split files already exist
if os.path.exists(train_path) and os.path.exists(test_path):
    print("Split files already exist. Skipping split operation.")
else:
    # Read the dataset
    df = pd.read_csv('../data/Bank_Transaction_Fraud_Detection.csv')
    
    # Perform 80-20 train-test split with stratification on target column
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['Is_Fraud']
    )
    
    # Save to CSV files in data folder
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data split completed successfully!")
    print(f"Training set: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Fraud cases in training: {train_df['Is_Fraud'].sum()}")
    print(f"Fraud cases in test: {test_df['Is_Fraud'].sum()}")