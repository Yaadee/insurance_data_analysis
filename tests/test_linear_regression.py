import unittest
import pandas as pd
from src.data_preparation.prepare_data import handle_missing_data, encode_categorical_data, feature_engineering, prepare_data
from src.machine_learning.linear_regression import train_linear_regression, evaluate_model

class TestLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_csv('data/datasets/historical_insurance_data.csv')
        df = handle_missing_data(df)
        categorical_columns = ['Gender', 'Province']
        df = encode_categorical_data(df, categorical_columns)
        df = feature_engineering(df)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = prepare_data(df, target_column='TotalClaims')

    def test_train_linear_regression(self):
        model = train_linear_regression(self.X_train, self.y_train)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        model = train_linear_regression(self.X_train, self.y_train)
        mse, r2 = evaluate_model(model, self.X_test, self.y_test)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)

if __name__ == '__main__':
    unittest.main()
