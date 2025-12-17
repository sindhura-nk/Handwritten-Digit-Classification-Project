import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

# Set the tab=>page title 
st.set_page_config(page_title="HandWritten Digits Prediction",layout='wide')

# Set the page tile
st.title("HandWritten Digits Prediction - An Image Classification Project")
st.subheader("By Sindhura Kuntamukkula")
st.subheader("Upload the handwritten digit image(black and white) and click on Predict button to view the predicted results")

# Take all the required inputs from the user
upload_image = st.file_uploader("Upload your Image here",type=["png", "jpg", "jpeg"] )

# Preprocess image
def preprocess_image(uploaded_file):
    """
    Converts Streamlit uploaded file to model-ready ndarray
    """
    # Read image using PIL
    image = Image.open(uploaded_file)

    # Convert to grayscale
    image = image.convert("L")

    # Convert PIL image to NumPy array
    img = np.array(image)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Background check & invert if required
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize
    img = img.astype("float32") / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    return img



# Provide a button for user to click and get the predictions
submit = st.button('Predict the results here')

# Load the keras model files: model
model = load_model(r'/workspaces/Handwritten-Digit-Classification-Project/DigitsPrediction.keras')

# What should happen when user clicks on submit button
if submit:
     if upload_image is not None:
        img_array = preprocess_image(upload_image)

        preds = model.predict(img_array)
        final_pred = np.argmax(preds, axis=1)

        st.success(f"✅ Predicted Digit: **{final_pred[0]}**")
else:
    st.warning("⚠️ Please upload an image first.")
