import matplotlib.pyplot as plt
import seaborn as sns

def plot_p_values(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y)
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

