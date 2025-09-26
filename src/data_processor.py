import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def load_data(file_path):
    """Loads a CSV dataset."""
    return pd.read_csv(file_path)

def generate_molecular_descriptors(smiles_list):
    """
    Generates molecular descriptors (features) from a list of SMILES strings.
    
    Args:
        smiles_list (list): A list of SMILES strings.
        
    Returns:
        pd.DataFrame: A DataFrame of molecular descriptors.
    """
    descriptor_names = [d[0] for d in Descriptors.descList]
    
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Calculate a subset of descriptors to start with
            # You will need to select the most relevant ones for your model
            row = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)]
            descriptors.append(row)
        else:
            descriptors.append([None] * 3) # Handle invalid SMILES
            
    return pd.DataFrame(descriptors, columns=['MolWt', 'MolLogP', 'TPSA'])

def preprocess_data(df):
    """
    Cleans and preprocesses the raw data.
    
    Args:
        df (pd.DataFrame): The raw DataFrame.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Drop rows with missing values in the SMILES column
    df.dropna(subset=['SMILES'], inplace=True)
    
    # Generate molecular features
    features_df = generate_molecular_descriptors(df['SMILES'])
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    # Drop rows where descriptor generation failed
    df.dropna(inplace=True)
    
    # You might also perform one-hot encoding for categorical tags here
    
    return df

if __name__ == '__main__':
    # Example usage
    raw_df = load_data('data/raw_data.csv')
    processed_df = preprocess_data(raw_df)
    processed_df.to_csv('data/processed_data.csv', index=False)
    print("Data processing complete.")