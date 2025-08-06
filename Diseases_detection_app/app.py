import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gradio as gr
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import json
from PIL import Image

# Load model
MODEL_PATH = "best_model_densenet121.h5"

model = load_model(MODEL_PATH)

# Load class labels
CLASSES_PATH = "Diseases_Detection_app/classes.json"
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH) as f:
        raw_classes = json.load(f)
        class_names = {int(v): k for k, v in raw_classes.items()}
else:
    raise FileNotFoundError(f"Class file not found at {CLASSES_PATH}")

# Prediction function
def predict_disease(img: Image.Image):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    class_name = class_names.get(predicted_class, "Unknown")
    return f"🔍 Predicted Disease: **{class_name}**"

# Gradio Interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs=gr.Markdown(),
    title="🌿 Plant Disease Detection App",
    description="Upload a plant leaf image to detect the disease.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch(share=True)

