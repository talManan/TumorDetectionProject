import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
log_reg = LogisticRegression(max_iter=2000, solver='saga')
svm = SVC()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
print("Training Logistic Regression...")
log_reg.fit(X_train, y_train)
print("Training SVM...")
svm.fit(X_train, y_train)
print("Training Random Forest...")
rf.fit(X_train, y_train)

# Predict and evaluate
models = {'Logistic Regression': log_reg, 'SVM': svm, 'Random Forest': rf}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model and scaler
joblib.dump(rf, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nBest model (Random Forest) and scaler saved as 'best_model.pkl' and 'scaler.pkl'")