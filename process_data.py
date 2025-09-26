import pandas as pd
import os

# Define the folder where your data files are located
data_folder = 'data'

# A list to hold the individual DataFrames
taste_data_frames = []

# Define a mapping of known inconsistent column names to a standard set
column_map = {
    'name': 'name',
    'Name': 'name',
    'canonical_smiles': 'smiles',
    'canonical SMILES': 'smiles',
    'SMILES': 'smiles',
    'Taste': 'taste',
    'Class taste': 'taste'
}

print("--- Starting Data Processing ---")

# Process the bitter and sweet files and store them in a list
bitter_sweet_files = ['bitter-test.csv', 'bitter-train.csv', 'sweet-test.csv', 'sweet-train.csv']
for file_name in bitter_sweet_files:
    try:
        df = pd.read_csv(os.path.join(data_folder, file_name))
        print(f"Successfully loaded '{file_name}' with shape {df.shape}")
        
        # Standardize column names
        df.columns = [column_map.get(col, col) for col in df.columns]
        
        # Add a 'taste' column if it's not present
        if 'taste' not in df.columns:
            if 'sweet' in file_name:
                df['taste'] = 'Sweet'
            elif 'bitter' in file_name:
                df['taste'] = 'Bitter'
            
        # Select the standardized columns and append to the list
        if all(col in df.columns for col in ['name', 'smiles', 'taste']):
            taste_data_frames.append(df[['name', 'smiles', 'taste']])
        else:
            print(f"Skipping '{file_name}' due to missing essential columns.")
            
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{file_name}': {e}")

# Process the ChemTastesDB database file
try:
    df_chem = pd.read_csv(os.path.join(data_folder, 'ChemTastesDB_database.csv'))
    print(f"\nSuccessfully loaded 'ChemTastesDB_database.csv' with shape {df_chem.shape}")
    
    # Rename columns using the mapping
    df_chem.columns = [column_map.get(col, col) for col in df_chem.columns]

    # Clean the 'taste' column
    df_chem['taste'] = df_chem['taste'].str.split('/').str[0].str.strip()
    
    # Select the standardized columns and append to the list
    if all(col in df_chem.columns for col in ['name', 'smiles', 'taste']):
        taste_data_frames.append(df_chem[['name', 'smiles', 'taste']])
    else:
        print(f"Skipping 'ChemTastesDB_database.csv' due to missing essential columns.")
        
except FileNotFoundError:
    print("File not found: ChemTastesDB_database.csv")
except Exception as e:
    print(f"An error occurred while processing 'ChemTastesDB_database.csv': {e}")


# Concatenate all DataFrames with a standardized schema
if taste_data_frames:
    # Use this alternative to pd.concat to avoid the error
    combined_df = pd.DataFrame(columns=['name', 'smiles', 'taste'])
    for df in taste_data_frames:
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    print("\n--- Basic Data Cleaning ---")
    
    # Drop rows with any missing 'name' or 'smiles' data
    initial_shape = combined_df.shape
    combined_df.dropna(subset=['name', 'smiles'], inplace=True)
    print(f"Dropped {initial_shape[0] - combined_df.shape[0]} rows with missing values.")

    # Remove duplicates
    initial_shape = combined_df.shape
    combined_df.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
    print(f"Removed {initial_shape[0] - combined_df.shape[0]} duplicate rows.")
    
    # Clean 'taste' column values
    combined_df['taste'] = combined_df['taste'].str.strip().str.lower()
    print("\nUnique values in 'taste' column after standardization:")
    print(combined_df['taste'].unique())

    # Save the cleaned data to a new CSV file
    output_file_path = os.path.join(data_folder, 'processed_taste_data.csv')
    combined_df.to_csv(output_file_path, index=False)
    
    print("\n--- Processing Complete! ---")
    print(f"Cleaned taste data saved to: '{output_file_path}'")
    print("Final DataFrame head:")
    print(combined_df.head())
else:
    print("\nNo taste-related files were loaded for processing. Please check your data folder.")