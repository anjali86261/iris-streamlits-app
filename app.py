import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('iris_model.pkl', 'rb'))

st.title("🌸 Iris Flower Prediction App")

# Input fields
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Prediction button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Flower: {species[prediction[0]]}")