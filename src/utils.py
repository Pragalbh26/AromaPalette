import joblib
import pandas as pd
import os

def load_model(model_path):
    """
    Loads a machine learning model from a given file path.

    Args:
        model_path (str): The file path to the saved model (.pkl).

    Returns:
        The loaded model object or None if loading fails.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_tags(tags_file_path):
    """
    Loads a list of aroma/flavor tags from a file.
    
    Args:
        tags_file_path (str): The file path to a text or CSV file containing the tags.
        
    Returns:
        list: A list of tags.
        
    Raises:
        FileNotFoundError: If the tags file does not exist.
    """
    if not os.path.exists(tags_file_path):
        raise FileNotFoundError(f"Tags file not found at: {tags_file_path}")
    
    tags = []
    with open(tags_file_path, 'r') as f:
        # Assuming one tag per line
        tags = [line.strip() for line in f if line.strip()]
        
    print(f"{len(tags)} tags loaded successfully from {tags_file_path}")
    return tags

# Example Usage
if __name__ == '__main__':
    # You would need to create these files for this example to work
    # with open('dummy_tags.txt', 'w') as f:
    #     f.write("Fruity\nFloral\nSpicy")
    
    # Load the tags
    try:
        loaded_tags = load_tags('dummy_tags.txt')
        print(f"Loaded tags: {loaded_tags}")
    except FileNotFoundError as e:
        print(e)
        
    # Example of how you would use load_model in another script
    # from utils import load_model
    # model_path = '../trained_models/fragrance_predictor.pkl'
    # my_model = load_model(model_path)
    # if my_model:
    #     # Do something with the model
    #     pass