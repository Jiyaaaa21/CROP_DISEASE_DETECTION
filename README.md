# 🌾 Crop Disease Detection using Deep Learning

> 📌 Developed by **Jyoti**, B.Tech AI Student  
> 🔍 Showcasing end-to-end AI skills in Deep Learning, Image Classification, and App Deployment.

---

## 📌 Overview

This project is a robust, AI-powered system designed to classify plant diseases from leaf images using advanced deep learning techniques. It detects **38 different plant diseases** with high precision using pre-trained CNN models. The best-performing model, **DenseNet121**, achieved **99.15% accuracy** and was deployed using **Streamlit** for real-time inference.

---

## ✅ Features

- 🔍 38-Class Disease Classification  
- 🤖 Transfer Learning with ResNet50, MobileNetV2, DenseNet121  
- 📈 Evaluation Visuals: Confusion Matrix, Reports, Accuracy Graphs  
- 🌐 Deployed Streamlit App for Real-Time Predictions  
- 🧠 Trained on 100K+ Images under variable lighting conditions

---

## 🎯 Highlights

- 🔬 **End-to-End ML Pipeline**: Data → Modeling → Evaluation → Deployment  
- 🧪 **Transfer Learning & Fine-Tuning**  
- 📊 **Classification Reports + Confusion Matrices** saved as PNGs  
- 💻 **Streamlit App** to detect diseases by uploading a leaf image  
- 📦 Structured project for easy reproducibility and extension

---

## 📁 Project Structure

CROP_DISEASE_DETECTION/
│
├── data/ # Training, validation, test sets
├── notebooks/ # Jupyter notebooks for EDA, preprocessing, training
├── models/ # Trained models and accuracy plots
├── models_evaluation/ # Confusion matrix, classification reports
├── models_comparison/ # Accuracy comparison graphs
├── Diseases_detection_app/ # Streamlit deployment
│ ├── app.py # Streamlit code
│ ├── best_model_densenet121.h5
│ └── classes.json
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🧠 Skills Demonstrated

- ✅ Deep Learning using **TensorFlow / Keras**
- ✅ Transfer Learning, Fine-Tuning
- ✅ Data Preprocessing and Augmentation
- ✅ Image Classification with CNNs
- ✅ Deployment using **Streamlit**
- ✅ Evaluation using **Scikit-learn**, **Matplotlib**, **Seaborn**
- ✅ Git/GitHub version control & clean project structure

---

## 📊 Model Evaluation

| Model         | Validation Accuracy |
|---------------|---------------------|
| **DenseNet121** | **99.15% ✅**        |
| ResNet50      | 98.72%              |
| MobileNetV2   | 97.86%              |

- 📌 DenseNet121 was selected for deployment based on its performance.

---

## 🚀 How to Run the App Locally

### 1. Clone the Repository

git clone https://github.com/Jiyaaaa21/CROP_DISEASE_DETECTION.git
cd CROP_DISEASE_DETECTION

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Run Streamlit App

cd Diseases_detection_app
streamlit run app.py

## 🖼️ Screenshots

### 📊 Confusion Matrix
![Confusion Matrix](models_evaluation\Evaluation_report\confusion_matrix_densenet121.png)

### 📊 Model Evaluation
![Models Comparison](models_comparison\model_comparison_metrics.png)

📦 Dataset
Dataset Name: New Plant Diseases Dataset (Augmented)

Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Contains over 87,000+ augmented images across 38 plant disease classes


🙋‍♀️ About Me
👩‍💻 Jyoti
🎓 B.Tech (Artificial Intelligence), 3rd Year
📫 Email: chandilajiya81@gmail.com
🔗 www.linkedin.com/in/jyotichandila
💡 Passionate about solving real-world problems in agriculture and healthcare using AI

⭐ This project reflects my strong interest in combining AI with agriculture to solve impactful, real-world problems. I am actively looking for opportunities to apply my skills in internships and research projects.

 