import scipy.stats as stats


def perform_ab_testing(data, feature):
    """
    Implement A/B Hypothesis Testing to evaluate risk differences across provinces, zip codes, and gender.
    
    Args:
        data (DataFrame): The dataset containing the relevant features.
        feature (str): The feature for which A/B testing is to be performed.
    """
    # Data Segmentation
    group_A = data[data['Group'] == 'A'][feature]
    group_B = data[data['Group'] == 'B'][feature]
    
    # Statistical Testing
    if data[feature].dtype == 'object':
        # For categorical data
        chi2_stat, p_value = stats.chisquare(group_A.value_counts(), group_B.value_counts())
    else:
        # For numerical data
        t_stat, p_value = stats.ttest_ind(group_A, group_B)
    
    # Analyze the p-value
    significance_level = 0.05
    if p_value < significance_level:
        print(f"There is a statistically significant difference in {feature} between the groups.")
    else:
        print(f"There is no statistically significant difference in {feature} between the groups.")

# Example usage:
# perform_ab_testing(data, 'Province')
# perform_ab_testing(data, 'ZipCode')
# perform_ab_testing(data, 'Gender')
