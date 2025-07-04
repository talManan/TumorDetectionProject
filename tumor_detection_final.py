import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset and split
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_model.pkl')

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy:.2f}")
print("Final Classification Report:\n", classification_report(y_test, y_pred))

# Save 
with open('results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
print("Results saved to 'results.txt'")
