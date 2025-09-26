import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import os
import sys
import cirpy  # Library to resolve chemical names

# --- Load Model and Assets ---
# This section loads your pre-trained model and data, ensuring
# that all predictions are based on your work.
try:
    model = joblib.load(os.path.join('data', 'random_forest_model.joblib'))
    le = joblib.load(os.path.join('data', 'label_encoder.joblib'))
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv'))
    feature_columns = X_train.columns
except FileNotFoundError as e:
    print(f"Error: A required file was not found. {e}")
    print("Please ensure you have run the 'train_rf.py' and 'data_prep.py' scripts.")
    sys.exit(1)

print("--- Welcome to the Interactive Taste Predictor! ---")
print("This tool predicts the taste of a compound from its name or SMILES string.")
print("Enter a compound name (e.g., 'Caffeine', 'Aspartame') or a SMILES string.")
print("Type 'exit' or 'quit' to close the program.")

# --- Main Prediction Loop ---
while True:
    # --- Step 1: Get and Interpret User Input ---
    user_input = input("\nEnter compound name or SMILES: ").strip()

    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the predictor. Goodbye!")
        break

    smiles_to_predict = None
    original_input_name = user_input

    # --- Step 2: Resolve Name or Use as SMILES ---
    # First, try to resolve the input as a chemical name online.
    print(f"--> Searching for '{original_input_name}' as a chemical name...")
    smiles_to_predict = cirpy.resolve(user_input, 'smiles')

    if smiles_to_predict:
        print(f"--> Successfully resolved to SMILES: {smiles_to_predict}")
    else:
        # If resolution fails, assume the input is already a SMILES string.
        print("--> Could not resolve name. Treating input as a SMILES string.")
        smiles_to_predict = user_input

    # --- Step 3: Generate Features for the Compound ---
    mol = Chem.MolFromSmiles(smiles_to_predict)

    if mol:
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol)
        }
        # Create a DataFrame from the descriptors, ensuring the column order is correct.
        new_data = pd.DataFrame([descriptors], columns=feature_columns)
        
        # --- Step 4: Make a Prediction with Your Model ---
        prediction_encoded = model.predict(new_data)

        # --- Step 5: Decode and Display the Prediction ---
        predicted_taste = le.inverse_transform(prediction_encoded)
        
        print("\n--- Prediction Result ---")
        print(f"The predicted taste for '{original_input_name}' is: {predicted_taste[0].upper()}")
    else:
        # This block runs if the final SMILES string was invalid.
        print(f"\nError: Input '{original_input_name}' could not be resolved to a valid molecule.")
        print("Please check the name or SMILES string and try again.")
    
    print("-" * 30) # Separator for the next prediction

