import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the prepared training data
data_folder = 'data'
X_train = pd.read_csv(os.path.join(data_folder, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(data_folder, 'y_train.csv')).squeeze()

print("--- Starting Hyperparameter Tuning with GridSearchCV ---")

# Define the model to tune
model = RandomForestClassifier(random_state=42)

# Define the grid of parameters to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("\n--- Hyperparameter Tuning Complete ---")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# You can now use grid_search.best_estimator_ for final predictions
# on your test set