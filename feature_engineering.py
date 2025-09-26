import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# Load the processed data
data_folder = 'data'
processed_file = os.path.join(data_folder, 'processed_taste_data.csv')
df = pd.read_csv(processed_file)

print("--- Starting Feature Engineering with RDKit ---")
print(f"Loaded data with shape: {df.shape}")

# Create an empty list to store the calculated descriptors
descriptors_list = []

# Loop through each row of the dataframe
for index, row in df.iterrows():
    smiles = row['smiles']
    
    # Try to create a molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    
    # If the molecule is valid, calculate the descriptors
    if mol:
        desc_dict = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),  # Hydrogen Bond Donors
            'HBA': Descriptors.NumHAcceptors(mol), # Hydrogen Bond Acceptors
            'TPSA': Descriptors.TPSA(mol)  # Topological Polar Surface Area
            # Add more descriptors here if you want
        }
        descriptors_list.append(desc_dict)

    else:
        # If the SMILES string is invalid, append a dictionary of NaNs
        desc_dict = {
            'MW': float('nan'),
            'LogP': float('nan'),
            'HBD': float('nan'),
            'HBA': float('nan'),
            'TPSA': float('nan')
        }
        descriptors_list.append(desc_dict)

# Create a new DataFrame from the descriptors list
descriptors_df = pd.DataFrame(descriptors_list)

# Concatenate the new descriptors DataFrame with the original DataFrame
final_df = pd.concat([df, descriptors_df], axis=1)

# Drop any rows where descriptor calculation failed
final_df.dropna(subset=['MW'], inplace=True)

# Save the final dataframe with features
output_file = os.path.join(data_folder, 'data_with_features.csv')
final_df.to_csv(output_file, index=False)

print("\n--- Feature Engineering Complete! ---")
print(f"Features saved to: '{output_file}'")
print(f"Final data shape: {final_df.shape}")
print("\nFinal DataFrame head:")
print(final_df.head())