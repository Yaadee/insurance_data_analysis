import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

def read_data(file_path):
    """Reads the CSV file into a DataFrame."""
    df = pd.read_csv(file_path, low_memory=False)
    return df

def preprocess_data(df):
    """Preprocesses the DataFrame."""
    # Convert 'TransactionMonth' column to date format
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])

    # Convert boolean columns to integers (0 and 1)
    boolean_columns = df.select_dtypes(include=['bool']).columns
    for column in boolean_columns:
        df[column] = df[column].astype(int)

    # Perform label encoding for categorical columns
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    return df, label_encoders

def conduct_t_test(group_a, group_b):
    """Conducts t-test for two groups."""
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    return p_value

def analyze_provinces(df):
    """Analyzes risk differences across provinces."""
    provinces = df['Province'].unique()
    for province in provinces:
        group_a = df[df['Province'] == province]['TotalClaims']
        group_b = df[df['Province'] != province]['TotalClaims']
        p_value = conduct_t_test(group_a, group_b)
        print(f'Province {province}: p-value={p_value}')
        if p_value < 0.05:
            print(f'Reject the null hypothesis for province {province}.')
        else:
            print(f'Fail to reject the null hypothesis for province {province}.')

def analyze_zipcodes(df):
    """Analyzes risk differences between zip codes."""
    zipcodes = df['PostalCode'].unique()
    for zipcode in zipcodes:
        group_a = df[df['PostalCode'] == zipcode]['TotalClaims']
        group_b = df[df['PostalCode'] != zipcode]['TotalClaims']
        p_value = conduct_t_test(group_a, group_b)
        print(f'Zip Code {zipcode}: p-value={p_value}')
        if p_value < 0.05:
            print(f'Reject the null hypothesis for zip code {zipcode}.')
        else:
            print(f'Fail to reject the null hypothesis for zip code {zipcode}.')

def analyze_margin_differences(df):
    """Analyzes margin (profit) differences between zip codes."""
    zipcodes = df['PostalCode'].unique()
    for zipcode in zipcodes:
        group_a = df[df['PostalCode'] == zipcode]['TotalPremium'] - df[df['PostalCode'] == zipcode]['TotalClaims']
        group_b = df[df['PostalCode'] != zipcode]['TotalPremium'] - df[df['PostalCode'] != zipcode]['TotalClaims']
        p_value = conduct_t_test(group_a, group_b)
        print(f'Margin difference for zip code {zipcode}: p-value={p_value}')
        if p_value < 0.05:
            print(f'Reject the null hypothesis for margin difference in zip code {zipcode}.')
        else:
            print(f'Fail to reject the null hypothesis for margin difference in zip code {zipcode}.')

def analyze_gender_differences(df):
    """Analyzes risk differences between Women and Men."""
    group_women = df[df['Gender'] == 'Female']['TotalClaims']
    group_men = df[df['Gender'] == 'Male']['TotalClaims']
    t_stat, p_value = stats.ttest_ind(group_women, group_men)
    print(f'Gender risk difference: p-value={p_value}')
    if p_value < 0.05:
        print('Reject the null hypothesis for gender risk difference.')
    else:
        print('Fail to reject the null hypothesis for gender risk difference.')
