import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importances, title="Feature Importance"):
    features, importance_values = zip(*importances)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_values, y=features)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def plot_model_performance(models, mse_scores, r2_scores, training_times):
    plt.figure(figsize=(12, 6))
    
    # MSE Scores
    plt.subplot(1, 3, 1)
    sns.barplot(x=models, y=mse_scores)
    plt.title("MSE Scores")
    plt.xlabel("Model")
    plt.ylabel("MSE")
    
    # R2 Scores
    plt.subplot(1, 3, 2)
    sns.barplot(x=models, y=r2_scores)
    plt.title("R2 Scores")
    plt.xlabel("Model")
    plt.ylabel("R2")

    # Training Times
    plt.subplot(1, 3, 3)
    sns.barplot(x=models, y=training_times)
    plt.title("Training Times")
    plt.xlabel("Model")
    plt.ylabel("Time (s)")

    plt.tight_layout()
    plt.show()
