from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import pandas as pd
import joblib
import os
import cirpy
from collections import Counter
import base64
from io import BytesIO

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load ML Model and necessary files ---
try:
    model = joblib.load(os.path.join('data', 'random_forest_model.joblib'))
    le = joblib.load(os.path.join('data', 'label_encoder.joblib'))
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv'))
    feature_columns = X_train.columns
except FileNotFoundError:
    print("ERROR: Make sure model files are in the 'data' directory.")
    model, le, feature_columns = None, None, None

# --- Predefined list of perfume ingredients for mixer ---
perfume_ingredients = [
    "Vanillin", "Limonene", "Linalool", "Geraniol", "Menthol",
    "Eugenol", "Citronellol", "Alpha-Pinene", "Beta-Pinene",
    "Jasmine", "Rose", "Sandalwood", "Lavender"
]

# --- Helper Functions ---
def get_features(smiles_string):
    """Calculates molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol and feature_columns is not None:
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol)
        }
        return pd.DataFrame([descriptors], columns=feature_columns)
    return None

def get_molecule_image(smiles_string):
    """Generates a 2D image of a molecule from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    return None

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html', ingredients=perfume_ingredients)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction for a single molecule."""
    if not all([model, le, feature_columns]):
        return jsonify({'error': 'Model not loaded properly'}), 500

    identifier = request.json.get('identifier')
    if not identifier:
        return jsonify({'error': 'No identifier provided'}), 400

    smiles = cirpy.resolve(identifier, 'smiles')
    if not smiles:
        smiles = identifier  # Assume input was already SMILES

    features = get_features(smiles)
    if features is None:
        return jsonify({'error': f'Invalid SMILES or name: {identifier}'}), 400

    prediction_encoded = model.predict(features)
    predicted_taste = le.inverse_transform(prediction_encoded)[0]
    image_data = get_molecule_image(smiles)

    return jsonify({
        'prediction': predicted_taste.capitalize(),
        'smiles': smiles,
        'image': image_data
    })

@app.route('/blend', methods=['POST'])
def blend():
    """Handles prediction for a blend of molecules."""
    if not all([model, le]):
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    ingredients = request.json.get('ingredients')
    if not ingredients:
        return jsonify({'dominant_scent': 'None'})

    predictions = []
    for name in ingredients:
        smiles = cirpy.resolve(name, 'smiles')
        if smiles:
            features = get_features(smiles)
            if features is not None:
                pred_encoded = model.predict(features)
                pred_taste = le.inverse_transform(pred_encoded)[0]
                predictions.append(pred_taste)

    if not predictions:
        return jsonify({'dominant_scent': 'Mix is Invalid'})

    dominant_scent = Counter(predictions).most_common(1)[0][0]
    return jsonify({'dominant_scent': dominant_scent.capitalize()})

@app.route('/ingredients', methods=['GET'])
def get_ingredients():
    """Returns the list of available ingredients."""
    return jsonify(perfume_ingredients)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

