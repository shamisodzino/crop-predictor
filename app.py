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
    N = st.sidebar.number_input("Nitrogen (N) (mg/kg)", 0, 1000, 120)
    P = st.sidebar.number_input("Phosphorus (P) (mg/kg)", 0, 1000, 20)
    K = st.sidebar.number_input("Potassium (K) (mg/kg)", 0, 1000, 300)
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
st.header("The most suitable crop to grow are (in Ascending order): ")
fig, ax = plt.subplots()
ax.barh(proba_df['Crop'], proba_df['Probability'])
ax.set_xlabel('Probability')
ax.set_ylabel('Crop')
ax.set_title('Predicted Crop Probabilities')

# Display the bar chart in the main app
st.pyplot(fig)