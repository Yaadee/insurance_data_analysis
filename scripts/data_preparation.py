import pandas as pd
import numpy as np 

# # Load the dataset from the text file
# data_txt = "/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/MachineLearningRating_v3.txt" 
# df = pd.read_csv(data_txt, delimiter='|', low_memory=False)
# # Save the dataset as a CSV file
# data_csv = "/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/historical_insurance_data.csv"
# df.to_csv(data_csv, index=False)


# # After data is converted to csv 


# Load the dataset
data = pd.read_csv('data/datasets/historical_insurance_data.csv', low_memory=False)

# Calculate descriptive statistics
summary_stats = data[['TotalPremium', 'TotalClaims']].describe()
print(summary_stats)


'''The TotalPremium column has a mean value of approximately 61.91 and a standard deviation of 230.28, indicating some variability in premium amounts.
The TotalClaims column has a mean value of approximately 64.86 and a standard deviation of 2384.08, suggesting higher variability in claim amounts compared to premiums.
'''
#  find if there are missing values check quality of data and replace with their mean value

missing_values = data.isnull().sum()

print("Number of missing values\n")

print(missing_values)

numerical_columns = data.select_dtypes(include=np.number).columns


# Impute missing values for numerical features
for col in data.select_dtypes(include='number').columns:
    data[col] = data[col].fillna(data[col].mean())


# Impute missing values for categorical features
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna('Unknown')





missing_values = data.isnull().sum()

print("Number of missing values after data cleaning\n")

print(missing_values)






# # Feature Engineering
# # Extract year and month from TransactionMonth
# df['TransactionYear'] = pd.to_datetime(df['TransactionMonth']).dt.year
# df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth']).dt.month

# # One-hot encode all categorical features
# for col in df.select_dtypes(include='object').columns:
#     df = pd.get_dummies(df, columns=[col], drop_first=True)

# # Label encoding for categorical features (example: LegalType)
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# df['LegalType'] = label_encoder.fit_transform(df['LegalType'])

# # Save the preprocessed data
# df.to_csv("preprocessed_data.csv", index=False)