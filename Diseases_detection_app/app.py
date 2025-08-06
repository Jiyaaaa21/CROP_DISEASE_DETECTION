import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import streamlit as st
from keras.models import load_model
from keras.utils import img_to_array
import json
from PIL import Image

# Load model
MODEL_PATH = "best_model_densenet121.h5"
model = load_model(MODEL_PATH)

# Load class labels
CLASSES_PATH = "classes.json"
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH) as f:
        raw_classes = json.load(f)
        class_names = {int(v): k for k, v in raw_classes.items()}
else:
    st.error(f"Class file not found at {CLASSES_PATH}")
    st.stop()

# App title
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a **plant leaf image** to detect the disease.")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    class_name = class_names.get(predicted_class, "Unknown")

    st.markdown(f"🔍 **Predicted Disease:** `{class_name}`")


