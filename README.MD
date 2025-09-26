AI Perfumist & Flavor Designer
This is an interactive web application that uses a machine learning model to predict the flavor and aroma profiles of chemical molecules. It also includes a creative "Perfume Mixer" mode to design custom scent blends.

This project demonstrates a unique, cross-domain application of machine learning, combining chemistry with a tangible, creative user experience.

Features
Single Molecule Analyzer:

Input a molecule's SMILES string.

View the 2D structure of the molecule.

Get a prediction for its primary aroma/flavor tag (e.g., Fruity, Floral, Spicy).

Visualize the full sensory profile on a dynamic "Aroma Wheel" radar chart.

Mix Your Own Perfume:

Select from a list of common fragrance ingredients (like Vanillin, Limonene, etc.).

The application predicts the profile for each ingredient and computationally "blends" them.

View the resulting aroma profile of your custom blend on a new Aroma Wheel.

How to Run the Application
1. Prerequisites
Make sure you have Python installed. You will also need to install the required libraries.

2. Setup
Clone the repository or place all your Python script files (.py) in a single project folder.

Create a data sub-directory inside your project folder.

Place your trained model files inside the data directory. You must have the following files:

data/random_forest_model.joblib

data/label_encoder.joblib

data/X_train.csv (This is needed to ensure the feature columns are correct)

Install dependencies from your requirements.txt file:

pip install -r requirements.txt

3. Launch the App (Step-by-Step)
Follow these steps exactly in your terminal (Command Prompt or PowerShell).

Navigate to your project folder using the cd command.

cd C:\Users\slash\OneDrive\Desktop\AromaPallete

Activate your virtual environment. This is a critical step.

.\venv\Scripts\activate

After running this, you should see (venv) at the beginning of your terminal prompt.

Verify your location. Run the dir command to make sure you see app.py in the list of files.

dir

Run the Streamlit app. Now that you are in the correct folder with the environment active, run the app.

streamlit run app.py

Your web browser should automatically open with the application running!

4. Troubleshooting
Error: can't open file 'app.py': [Errno 2] No such file or directory

What it means: You are trying to run the streamlit command from the wrong folder, or your virtual environment is not active.

How to fix it: Go back to the "Launch the App (Step-by-Step)" section and follow the steps in that exact order. The most common mistakes are:

Forgetting to cd into the AromaPallete directory first.

Forgetting to run .\venv\Scripts\activate to activate the environment before trying to run the app.