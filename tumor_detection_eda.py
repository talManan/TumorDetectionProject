import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Print the basic information
print("Dataset Shape:", X.shape)
print("\nFeature Names:\n", data.feature_names)
print("\nClass Distribution:\n", pd.Series(y).value_counts())

# Plot the feature distributions
plt.figure(figsize=(12, 6))
sns.histplot(X['mean radius'], kde=True, label='Mean Radius')
sns.histplot(X['mean texture'], kde=True, label='Mean Texture')
plt.legend()
plt.title('Feature Distributions')
plt.savefig('feature_distributions.png')
plt.show()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()
