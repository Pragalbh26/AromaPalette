import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the prepared training and testing data
data_folder = 'data'
X_train = pd.read_csv(os.path.join(data_folder, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(data_folder, 'y_train.csv'))
X_test = pd.read_csv(os.path.join(data_folder, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_folder, 'y_test.csv'))

# Convert y_train and y_test from DataFrame to Series for compatibility
y_train = y_train.squeeze()
y_test = y_test.squeeze()

print("--- Starting Model Training ---")

# Step 1: Initialize and train the model
model = LogisticRegression(max_iter=1000)
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

print("\n--- Process Complete! ---")