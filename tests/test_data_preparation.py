import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the text file
data_txt = "/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/MachineLearningRating_v3.txt"
df = pd.read_csv(data_txt, delimiter='|', low_memory=False)
# Save the dataset as a CSV file
data_csv = "/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/historical_insurance_data.csv"
df.to_csv(data_csv, index=False)

print(df.describe())
print(df.dtypes)

# Load the dataset
def loaddata():
    data = pd.read_csv('/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/historical_insurance_data.csv', low_memory=False)
    return data

data = loaddata()

# Data preparation functions
def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    """
    missing_values = data.isnull().sum()
    print("Number of missing values:\n")
    print(missing_values)
    
    # Impute missing values for numerical features
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].mean())
    
    # Impute missing values for categorical features
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].fillna('N/A')
    
    # Handling missing values specifically for 'NumberOfVehiclesInFleet' column
    if 'NumberOfVehiclesInFleet' in data.columns:
        data['NumberOfVehiclesInFleet'] = data['NumberOfVehiclesInFleet'].fillna(data['NumberOfVehiclesInFleet'].mean())
    
    missing_values = data.isnull().sum()
    print("Number of missing values after data cleaning:\n")
    print(missing_values)

def feature_engineering(data):
    """
    Perform feature engineering on the dataset.
    """
    # Create new feature by combining existing features
    if 'PolicyAmount' in data.columns and 'PolicyCount' in data.columns:
        data['TotalPolicyAmount'] = data['PolicyAmount'] * data['PolicyCount']
    
    # Extract year and month from a date column
    if 'TransactionDate' in data.columns:
        data['TransactionYear'] = pd.to_datetime(data['TransactionDate']).dt.year
        data['TransactionMonth'] = pd.to_datetime(data['TransactionDate']).dt.month

    return data

def encode_categorical_data(data):
    """
    Encode categorical features in the dataset.
    """
    # Custom encoding for specific columns
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0, 'Not specified': 3, 'N/A': 2})
    
    # Automatically encode other categorical features
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        if col != 'Gender':  # Skip already encoded column
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])
    
    return data

# Handle missing values
handle_missing_values(data)

# Perform feature engineering
data = feature_engineering(data)

# Encode categorical data
data = encode_categorical_data(data)

# Calculate descriptive statistics for specific columns
summary_stats = data[['TotalPremium', 'TotalClaims']].describe()
print(summary_stats)

"""
The TotalPremium column has a mean value of approximately 61.91 and a standard deviation of 230.28, indicating some variability in premium amounts.
The TotalClaims column has a mean value of approximately 64.86 and a standard deviation of 2384.08, suggesting higher variability in claim amounts compared to premiums.
"""

# Save the preprocessed data
preprocessed_data_csv = "/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/preprocessed_data.csv"
data.to_csv(preprocessed_data_csv, index=False)

print(data.describe())
print(data.dtypes)
print(data.head())
