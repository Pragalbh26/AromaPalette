import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the prepared training and testing data
data_folder = 'data'
X_train = pd.read_csv(os.path.join(data_folder, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(data_folder, 'y_train.csv')).squeeze()
X_test = pd.read_csv(os.path.join(data_folder, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_folder, 'y_test.csv')).squeeze()

print("--- Starting Random Forest Model Training ---")

# Step 1: Initialize and train the Random Forest model
# We'll use 100 trees and a fixed random_state for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel training complete.")

# Step 2: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 3: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Step 4: Save the trained model for future use
model_output_path = os.path.join('data', 'random_forest_model.joblib')
joblib.dump(model, model_output_path)
print(f"\nModel saved to: '{model_output_path}'")