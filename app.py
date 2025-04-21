import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from care_guide import care_guide  # Import care guide dictionary

# Get current directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = f"{working_dir}/plant_disease_pred_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Set Streamlit page config
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(96, 96)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, float(np.max(predictions))


# Sidebar
st.sidebar.title("🌿 Upload & Predict")
uploaded_image = st.sidebar.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])
predict_button = st.sidebar.button('🔍 Classify')

# Main Page
st.title('🌿 Plant Disease Classifier with Care Guide')
st.markdown("<hr style='margin:10px 0; border-color:gray'>", unsafe_allow_html=True)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    st.image(image.resize((200, 200)), caption="Uploaded Image", use_container_width=False)

    if predict_button:
        prediction, confidence = predict_image_class(model, uploaded_image, class_indices)

        # Styled prediction box
        st.markdown(f"""
            <div style="background-color:#14532d;padding:10px;border-radius:10px;margin-bottom:20px">
            <h4 style="color:#a7f3d0; text-align:center;">🌿 Prediction: <span style="color:white;">{prediction}</span></h4>
            <p style="color:#bbf7d0; text-align:center;">Confidence: {confidence * 100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)


        # Plant care information
        if prediction in care_guide:
            st.markdown("""
                <h3 style='color:#c4b5fd;'>🩺 Plant Care Information</h3>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background-color:#1e293b;padding:15px;border-radius:10px;margin-top:10px">
                    <p style="color:#d1fae5"><b>🌱 Symptoms:</b> {care_guide[prediction]['symptoms']}</p>
                    <p style="color:#fecaca"><b>💊 Treatment:</b> {care_guide[prediction]['treatment']}</p>
                    <p style="color:#fef9c3"><b>🛡 Prevention:</b> {care_guide[prediction]['prevention']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No care information found for this disease.")

# Footer
st.markdown("<hr style='margin-top: 30px; border-color: gray;'>", unsafe_allow_html=True)
