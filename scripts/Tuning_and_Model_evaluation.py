import pandas as pd
from src.data_preparation.prepare_data import load_data, handle_missing_data, encode_categorical_data, feature_engineering, prepare_data
from src.machine_learning.linear_regression import train_linear_regression, evaluate_model as evaluate_lr
from src.machine_learning.random_forest import train_random_forest, evaluate_model as evaluate_rf, get_feature_importance as get_rf_importance, tune_random_forest
from src.machine_learning.xgboost import train_xgboost, evaluate_model as evaluate_xgb, get_feature_importance as get_xgb_importance, tune_xgboost
from src.utils.visualization import plot_feature_importance, plot_model_performance

# Load and prepare data
df = load_data('data/datasets/historical_insurance_data.csv')
df = handle_missing_data(df)
categorical_columns = ['Gender', 'Province']
df = encode_categorical_data(df, categorical_columns)
df = feature_engineering(df)
X_train, X_test, y_train, y_test = prepare_data(df, target_column='TotalClaims')

# Model lists
models = ["Linear Regression", "Random Forest", "XGBoost"]
mse_scores = []
r2_scores = []
training_times = []

# Linear Regression
lr_model, lr_time = train_linear_regression(X_train, y_train)
lr_mse, lr_r2 = evaluate_lr(lr_model, X_test, y_test)
mse_scores.append(lr_mse)
r2_scores.append(lr_r2)
training_times.append(lr_time)

# Random Forest
rf_model, rf_time = train_random_forest(X_train, y_train)
rf_mse, rf_r2 = evaluate_rf(rf_model, X_test, y_test)
rf_importance = get_rf_importance(rf_model, X_train.columns)
mse_scores.append(rf_mse)
r2_scores.append(rf_r2)
training_times.append(rf_time)
plot_feature_importance(rf_importance, title="Random Forest Feature Importance")

# Hyperparameter Tuning for Random Forest
best_rf_model, best_rf_params = tune_random_forest(X_train, y_train)
best_rf_mse, best_rf_r2 = evaluate_rf(best_rf_model, X_test, y_test)
mse_scores.append(best_rf_mse)
r2_scores.append(best_rf_r2)
plot_feature_importance(get_rf_importance(best_rf_model, X_train.columns), title="Best Random Forest Feature Importance")
print(f"Best Random Forest Params: {best_rf_params}")

# XGBoost
xgb_model, xgb_time = train_xgboost(X_train, y_train)
xgb_mse, xgb_r2 = evaluate_xgb(xgb_model, X_test, y_test)
xgb_importance = get_xgb_importance(xgb_model, X_train.columns)
mse_scores.append(xgb_mse)
r2_scores.append(xgb_r2)
training_times.append(xgb_time)
plot_feature_importance(xgb_importance, title="XGBoost Feature Importance")

# Hyperparameter Tuning for XGBoost
best_xgb_model, best_xgb_params = tune_xgboost(X_train, y_train)
best_xgb_mse, best_xgb_r2 = evaluate_xgb(best_xgb_model, X_test, y_test)
mse_scores.append(best_xgb_mse)
r2_scores.append(best_xgb_r2)
plot_feature_importance(get_xgb_importance(best_xgb_model, X_train.columns), title="Best XGBoost Feature Importance")
print(f"Best XGBoost Params: {best_xgb_params}")

# Plot model performance
models += ["Best Random Forest", "Best XGBoost"]
plot_model_performance(models, mse_scores, r2_scores, training_times)
