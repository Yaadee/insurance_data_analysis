import pandas as pd
import numpy as np 

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
    
    missing_values = data.isnull().sum()
    print("Number of missing values after data cleaning\n")
    print(missing_values)

    # Handling missing values specifically for 'NumberOfVehiclesInFleet' column
    data['NumberOfVehiclesInFleet'] = data['NumberOfVehiclesInFleet'].fillna(data['NumberOfVehiclesInFleet'].mean())

def feature_engineering(data):
    """
    Perform feature engineering on the dataset.
    """
    # Example feature engineering techniques
    # Create new feature by combining existing features
    data['TotalPolicyAmount'] = data['PolicyAmount'] * data['PolicyCount']
    
    # Example: Extract year and month from a date column
    data['TransactionYear'] = pd.to_datetime(data['TransactionDate']).dt.year
    data['TransactionMonth'] = pd.to_datetime(data['TransactionDate']).dt.month

    # Return the modified DataFrame
    return data

def encode_categorical_data(data):
    """
    Encode categorical features in the dataset.
    """
    # Example encoding techniques
    # One-hot encoding for categorical features
    data = pd.get_dummies(data, columns=['PolicyType', 'VehicleMake'], drop_first=True)
    
    # Example: Label encoding for a categorical feature
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data['ClaimStatusEncoded'] = label_encoder.fit_transform(data['ClaimStatus'])

    # Return the modified DataFrame
    return data



# Call the function to handle missing values
handle_missing_values(data)

# Calculate descriptive statistics
summary_stats = data[['TotalPremium', 'TotalClaims']].describe()
print(summary_stats)

'''The TotalPremium column has a mean value of approximately 61.91 and a standard deviation of 230.28, indicating some variability in premium amounts.
The TotalClaims column has a mean value of approximately 64.86 and a standard deviation of 2384.08, suggesting higher variability in claim amounts compared to premiums.
'''

# Save the preprocessed data
data.to_csv("preprocessed_data.csv", index=False)

print(data.describe())

print(data.dtypes)

print(data.head())


# if there there is uncleaned data
import pandas as pd
import numpy as np 

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
    
    missing_values = data.isnull().sum()
    print("Number of missing values after data cleaning\n")
    print(missing_values)

    # Handling missing values specifically for 'NumberOfVehiclesInFleet' column
    data['NumberOfVehiclesInFleet'] = data['NumberOfVehiclesInFleet'].fillna(data['NumberOfVehiclesInFleet'].mean())

def feature_engineering(data):
    """
    Perform feature engineering on the dataset.
    """
    # Example feature engineering techniques
    # Create new feature by combining existing features
    data['TotalPolicyAmount'] = data['PolicyAmount'] * data['PolicyCount']
    
    # Example: Extract year and month from a date column
    data['TransactionYear'] = pd.to_datetime(data['TransactionDate']).dt.year
    data['TransactionMonth'] = pd.to_datetime(data['TransactionDate']).dt.month

    # Return the modified DataFrame
    return data

def encode_categorical_data(data):
    """
    Encode categorical features in the dataset.
    """
    # Example encoding techniques
    # One-hot encoding for categorical features
    data = pd.get_dummies(data, columns=['PolicyType', 'VehicleMake'], drop_first=True)
    
    # Example: Label encoding for a categorical feature
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data['ClaimStatusEncoded'] = label_encoder.fit_transform(data['ClaimStatus'])

    # Return the modified DataFrame
    return data



# Call the function to handle missing values
handle_missing_values(data)

# Calculate descriptive statistics
summary_stats = data[['TotalPremium', 'TotalClaims']].describe()
print(summary_stats)

'''The TotalPremium column has a mean value of approximately 61.91 and a standard deviation of 230.28, indicating some variability in premium amounts.
The TotalClaims column has a mean value of approximately 64.86 and a standard deviation of 2384.08, suggesting higher variability in claim amounts compared to premiums.
'''

# Save the preprocessed data
data.to_csv("preprocessed_data.csv", index=False)

print(data.describe())

print(data.dtypes)

print(data.head())