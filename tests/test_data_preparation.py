import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prepare_data import handle_missing_data, encode_categorical_data, feature_engineering, prepare_data

class TestDataPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('/home/yadasa/Desktop/InsuranceDataAnalysis/data/datasets/model_data.csv', low_memory=False)

    def test_handle_missing_data(self):
        df_cleaned = handle_missing_data(self.df)
        self.assertFalse(df_cleaned.isnull().values.any())

    def test_encode_categorical_data(self):
        categorical_columns = ['Gender', 'Province']
        df_encoded = encode_categorical_data(self.df, categorical_columns)
        for col in categorical_columns:
            self.assertNotIn(col, df_encoded.columns)

    def test_feature_engineering(self):
        df_engineered = feature_engineering(self.df)
        self.assertIn('ClaimRatio', df_engineered.columns)

    def test_prepare_data_total_claims(self):
        df_cleaned = handle_missing_data(self.df)
        categorical_columns = ['Gender', 'Province']
        df_encoded = encode_categorical_data(df_cleaned, categorical_columns)
        df_final = feature_engineering(df_encoded)
        X_train, X_test, y_train, y_test = prepare_data(df_final, target_column='TotalClaims')
        self.assertEqual(len(X_train) + len(X_test), len(df_final) - 1)  # Minus target column

    def test_prepare_data_total_premium(self):
        df_cleaned = handle_missing_data(self.df)
        categorical_columns = ['Gender', 'Province']
        df_encoded = encode_categorical_data(df_cleaned, categorical_columns)
        df_final = feature_engineering(df_encoded)
        X_train, X_test, y_train, y_test = prepare_data(df_final, target_column='TotalPremium')
        self.assertEqual(len(X_train) + len(X_test), len(df_final) - 1)  # Minus target column

if __name__ == '__main__':
    unittest.main()
