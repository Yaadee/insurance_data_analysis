import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'ab_testing'))

# Import the conduct_t_test function from ab_hypothesis_testing.py
from ab_hypothesis_testing import conduct_t_test, stats

class TestABTesting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Construct the path to the CSV file using the parent directory
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'model_data.csv')
        cls.df = pd.read_csv(data_path)

    def test_conduct_t_test(self):
        group_a = self.df[self.df['Province'] == 'Gauteng']['TotalClaims']
        group_b = self.df[self.df['Province'] != 'Gauteng']['TotalClaims']
        p_value = conduct_t_test(group_a, group_b)
        self.assertIsInstance(p_value, float)

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

if __name__ == '__main__':
    unittest.main()
