import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained U-Net model
@st.cache_resource  # Cache model to avoid reloading on every run
def load_model():
    model = tf.keras.models.load_model('UNet_Model (1).keras')
    return model

unet_model = load_model()

# Image preprocessing function (same as training preprocessing)
def preprocess_image(image):
    image_resized = cv2.resize(image, (256, 256))  # Resize to match input size
    image_normalized = image_resized / 255.0  # Normalize pixel values
    return np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# Perform inference and return predicted mask
def predict_lane(image):
    preprocessed_image = preprocess_image(image)
    predicted_mask = unet_model.predict(preprocessed_image)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Binary mask
    return predicted_mask

# Streamlit UI for file upload
st.title("Lane Detection using U-Net")
st.write("Upload a road image to detect lanes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert PIL image to OpenCV format
    image = np.array(image)

    # Predict lane mask
    st.write("Detecting lanes...")
    predicted_mask = predict_lane(image)

    # Display the predicted mask
    st.image(predicted_mask * 255, caption='Predicted Lane Mask', use_column_width=True)
