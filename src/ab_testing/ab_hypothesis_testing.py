import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv("/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/model_data.csv", low_memory=False)

# Convert 'TransactionMonth' column to date format
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])

# Convert boolean columns to integers (0 and 1)
boolean_columns = df.select_dtypes(include=['bool']).columns
for column in boolean_columns:
    df[column] = df[column].astype(int)

# Identify numerical columns (excluding 'UnderwrittenCoverID', 'PolicyID', 'TotalClaims', and 'TotalPremium')
excluded_columns = ['UnderwrittenCoverID', 'PolicyID', 'TotalClaims', 'TotalPremium']
numerical_columns = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in excluded_columns]

# Identify remaining categorical columns that are still of object type
remaining_categorical_columns = df.select_dtypes(include=['object']).columns

# Perform label encoding for each remaining categorical column
label_encoders = {}
for column in remaining_categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))  # Convert to string in case of mixed types
    label_encoders[column] = le

# Print label encodings
for column, le in label_encoders.items():
    print(f"Label encoding for column '{column}':")
    for value, encoded_value in zip(le.classes_, le.transform(le.classes_)):
        print(f"{value} -> {encoded_value}")

# Verify data types
print(df.dtypes)

# Fix target values
target_columns = ['TotalPremium', 'TotalClaims', 'Gender']  # Include 'Gender' column
df = df[['Province', 'PostalCode'] + target_columns]

# Risk differences across provinces
provinces = df['Province'].unique()
for province in provinces:
    group_a = df[df['Province'] == province]['TotalClaims']
    group_b = df[df['Province'] != province]['TotalClaims']
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    print(f'Province {province}: p-value={p_value}')
    if p_value < 0.05:
        print(f'Reject the null hypothesis for province {province}.')
    else:
        print(f'Fail to reject the null hypothesis for province {province}.')

# Risk differences between zip codes
zipcodes = df['PostalCode'].unique()
for zipcode in zipcodes:
    group_a = df[df['PostalCode'] == zipcode]['TotalClaims']
    group_b = df[df['PostalCode'] != zipcode]['TotalClaims']
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    print(f'Zip Code {zipcode}: p-value={p_value}')
    if p_value < 0.05:
        print(f'Reject the null hypothesis for zip code {zipcode}.')
    else:
        print(f'Fail to reject the null hypothesis for zip code {zipcode}.')

# Margin (profit) differences between zip codes
for zipcode in zipcodes:
    group_a = df[df['PostalCode'] == zipcode]['TotalPremium'] - df[df['PostalCode'] == zipcode]['TotalClaims']
    group_b = df[df['PostalCode'] != zipcode]['TotalPremium'] - df[df['PostalCode'] != zipcode]['TotalClaims']
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    print(f'Margin difference for zip code {zipcode}: p-value={p_value}')
    if p_value < 0.05:
        print(f'Reject the null hypothesis for margin difference in zip code {zipcode}.')
    else:
        print(f'Fail to reject the null hypothesis for margin difference in zip code {zipcode}.')

# Risk differences between Women and Men
group_women = df[df['Gender'] == 'Female']['TotalClaims']
group_men = df[df['Gender'] == 'Male']['TotalClaims']
t_stat, p_value = stats.ttest_ind(group_women, group_men)
print(f'Gender risk difference: p-value={p_value}')
if p_value < 0.05:
    print('Reject the null hypothesis for gender risk difference.')
else:
    print('Fail to reject the null hypothesis for gender risk difference.')
