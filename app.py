import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

# Set the tab=>page title 
st.set_page_config(page_title="HandWritten Digits Prediction",layout='wide')

# Set the page tile
st.title("HandWritten Digits Prediction - An Image Classification Project")
st.subheader("By Sindhura Kuntamukkula")
st.subheader("Upload the handwritten digit image(black and white) and click on Predict button to view the predicted results")

# Take all the required inputs from the user
upload_image = st.file_uploader("Upload your Image here",accept_multiple_files=True )

# Provide a button for user to click and get the predictions
submit = st.button('Predict the results here')

# Load the keras model files: model
model = keras.load_model(model_path)

# What should happen when user clicks on submit button
if submit:
    
