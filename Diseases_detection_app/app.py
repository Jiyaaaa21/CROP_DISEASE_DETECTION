import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os  
from PIL import Image  

# Set page config
st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")

# Title
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a plant leaf image to detect the disease.")

# Load model
MODEL_PATH = r"D:\Crop_disease_detection\Diseases_detection_app\best_model_densenet121.h5"
model = load_model(MODEL_PATH)

# Load class labels
CLASSES_PATH = "Diseases_Detection_app/classes.json"  
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH) as f:
        raw_classes = json.load(f)
        class_names = {int(v): k for k, v in raw_classes.items()} 
else:
    st.error(f"❌ Could not find class file at: {CLASSES_PATH}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Processing...")


    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))

    # Display result
    if predicted_class in class_names:
        st.success(f"🔍 Prediction: **{class_names[predicted_class]}**")
    else:
        st.warning(f"⚠️ Prediction class index not found: {predicted_class}")
    