import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load the data with features
data_folder = 'data'
features_file = os.path.join(data_folder, 'data_with_features.csv')
df = pd.read_csv(features_file)

# Drop unnecessary columns like 'name' and 'smiles'
X = df.drop(columns=['name', 'smiles', 'taste'])
y = df['taste']

print("--- Starting Data Preparation for ML ---")

# Step 1: Label Encoding
# Convert categorical 'taste' labels into numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, os.path.join(data_folder, 'label_encoder.joblib'))
print("\nTaste labels encoded:")
for original, encoded in zip(le.classes_, range(len(le.classes_))):
    print(f"  '{original}' -> {encoded}")

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("\nData splitting complete.")
print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")

# Save the prepared data to files
X_train.to_csv(os.path.join(data_folder, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(data_folder, 'X_test.csv'), index=False)
pd.DataFrame(y_train, columns=['taste_encoded']).to_csv(os.path.join(data_folder, 'y_train.csv'), index=False)
pd.DataFrame(y_test, columns=['taste_encoded']).to_csv(os.path.join(data_folder, 'y_test.csv'), index=False)

print("\n--- Data Preparation Complete! ---")
print("Saved training and testing sets to the 'data' folder.")