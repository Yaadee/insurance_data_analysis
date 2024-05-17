import unittest
import pandas as pd
import numpy as np
import os

class TestInsuranceDataPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset for testing
        data_txt = "/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/MachineLearningRating_v3.txt"
        cls.df = pd.read_csv(data_txt, delimiter='|', low_memory=False)
        cls.data_csv = "/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/historical_insurance_data.csv"
        cls.df.to_csv(cls.data_csv, index=False)

    def setUp(self):
        # Load the dataset for each test
        self.data = pd.read_csv(self.data_csv, low_memory=False)

    def test_handle_missing_values(self):
        handle_missing_values(self.data)
        self.assertFalse(self.data.isnull().values.any(), "There are still missing values after handling missing values.")

    def test_feature_engineering(self):
        initial_columns = set(self.data.columns)
        data_fe = feature_engineering(self.data)
        expected_columns = initial_columns.union({'TotalPolicyAmount', 'TransactionYear', 'TransactionMonth'})
        self.assertTrue(expected_columns.issubset(data_fe.columns), "Feature engineering did not create the expected columns.")

    def test_encode_categorical_data(self):
        data_encoded = encode_categorical_data(self.data)
        self.assertIn('ClaimStatusEncoded', data_encoded.columns, "Categorical data encoding did not create 'ClaimStatusEncoded' column.")
        self.assertTrue(pd.api.types.is_integer_dtype(data_encoded['ClaimStatusEncoded']), "'ClaimStatusEncoded' column is not of integer type.")

    def test_descriptive_statistics(self):
        handle_missing_values(self.data)
        summary_stats = self.data[['TotalPremium', 'TotalClaims']].describe()
        self.assertIn('mean', summary_stats.index, "Descriptive statistics did not compute 'mean'.")
        self.assertIn('std', summary_stats.index, "Descriptive statistics did not compute 'std'.")

def handle_missing_values(data):
    """Handle missing values in the dataset."""
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].mean())
    
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].fillna('N/A')
    
    if 'NumberOfVehiclesInFleet' in data.columns:
        data['NumberOfVehiclesInFleet'] = data['NumberOfVehiclesInFleet'].fillna(data['NumberOfVehiclesInFleet'].mean())

def feature_engineering(data):
    """Perform feature engineering on the dataset."""
    data['TotalPolicyAmount'] = data['PolicyAmount'] * data['PolicyCount']
    data['TransactionYear'] = pd.to_datetime(data['TransactionDate']).dt.year
    data['TransactionMonth'] = pd.to_datetime(data['TransactionDate']).dt.month
    return data

def encode_categorical_data(data):
    """Encode categorical features in the dataset."""
    data = pd.get_dummies(data, columns=['PolicyType', 'VehicleMake'], drop_first=True)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data['ClaimStatusEncoded'] = label_encoder.fit_transform(data['ClaimStatus'])
    return data

if __name__ == '__main__':
    unittest.main()
