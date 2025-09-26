import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y):
    """
    Trains a RandomForestClassifier model.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.DataFrame): Target labels (flavor/aroma tags).
        
    Returns:
        RandomForestClassifier: The trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model

def save_model(model, file_path):
    """Saves a trained model to a file."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Loads a trained model from a file."""
    return joblib.load(file_path)

def predict_tags(model, features):
    """
    Predicts flavor/aroma tags for a given set of molecular features.
    
    Args:
        model: The trained machine learning model.
        features (pd.DataFrame): DataFrame of molecular features.
        
    Returns:
        list: The predicted tags.
    """
    predictions = model.predict(features)
    # The model will predict a list of tags. You will need to map these to human-readable labels.
    return predictions.tolist()

if __name__ == '__main__':
    # This is a placeholder example. You will need to import your actual data here.
    # For a real project, you would load data from the data_processor.py script.
    print("Model training script is ready.")