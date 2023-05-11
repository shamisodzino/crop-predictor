import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model from the saved pickle file
with open("random_forest_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

# Define the app layout
st.title("Crop Prediction App")

# Create a sidebar header and markdown text
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Change the parameters to your own test to see the results")

# Define a function to get user inputs from the sidebar
def user_input_features():
    # Get input values for each parameter from the user
    Rainfall = st.sidebar.number_input("Rainfall (mm)", 0, 2500, 800)
    Temp = st.sidebar.number_input("Temperature (°C)", 0, 50, 25)
    pH = st.sidebar.number_input("Soil pH", 0.0, 14.0, 7.0)
    N = st.sidebar.number_input("Nitrogen (N) (kg/ha)", 0, 1000, 120)
    P = st.sidebar.number_input("Phosphorus (P) (kg/ha)", 0, 1000, 20)
    K = st.sidebar.number_input("Potassium (K) (kg/ha)", 1, 1000, 300)
    Mn = st.sidebar.number_input("Manganese (Mn) (mg/kg)", 0, 1000, 20)
    Zn = st.sidebar.number_input("Zinc (Zn) (mg/kg)", 0, 1000, 8)
    Cu = st.sidebar.number_input("Copper (Cu) (mg/kg)", 0, 1000, 2)

    # Create a dictionary to store user input values
    data = {'Rainfall': Rainfall,
            'Temp': Temp,
            'pH': pH,
            'Nitrogen (N)': N,
            'phosphours (P)': P,
            'Pottasium (K)': K,
            'Manganese (Mn)': Mn,
            'Zinc (Zn)': Zn,
            'Copper (Cu)': Cu}

    # Convert the dictionary into a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input values and store them in a DataFrame
input_df = user_input_features()

# Display the user input values in the main app
st.header("User Input Parameters")
st.write(input_df)

# Calculate the probabilities using the trained model
proba = clf.predict_proba(input_df)
proba_df = pd.DataFrame(proba, columns=clf.classes_)
proba_df = proba_df.T.sort_values(by=0, ascending=False)
proba_df.reset_index(level=0, inplace=True)
proba_df.columns = ['Crop', 'Probability']

# Sort the probabilities in descending order
proba_df = proba_df.sort_values(by='Probability', ascending=True)

# Create a horizontal bar chart with the probabilities
fig, ax = plt.subplots()
ax.barh(proba_df['Crop'], proba_df['Probability'])
ax.set_xlabel('Probability')
ax.set_ylabel('Crop')
ax.set_title('Most suitable Crops for your land are (in ascending order):')

# Display the bar chart in the main app
st.pyplot(fig)
# Add glossary section

st.header("About")

st.markdown("This Crop Recommendation App is a machine learning-powered web application that helps farmers and agronomists make informed decisions on which crops to grow based on specific soil and environmental conditions. The app takes input parameters such as rainfall, temperature, soil pH, and nutrient levels, and predicts the most suitable crops")
st.header("Glossary")

# Define the glossary terms and their explanations
glossary = {
    "kg/ha": "Kilograms per hectare, a unit of measurement for fertilizer application rate",
    "mg/kg": "Milligrams per kilogram, a unit of measurement for soil nutrient concentration",
    "pH": "A measure of the acidity or alkalinity of the soil",
    "C": "Degrees Celsius, a unit of measurement for temperature",
    "mm": "Millimeters, a unit of measurement for rainfall",
    "Cu": "Copper, a micronutrient essential for plant growth",
    "K": "Potassium, a macronutrient essential for plant growth and development",
    "P": "Phosphorus, a macronutrient essential for plant growth and development",
    "N": "Nitrogen, a macronutrient essential for plant growth and development",
    "Mn": "Manganese, a micronutrient essential for plant growth"
}

# Display the glossary terms and their explanations
for term, definition in glossary.items():
    st.write(f"**{term}:** {definition}")

