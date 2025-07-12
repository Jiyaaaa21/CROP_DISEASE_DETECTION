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
├── data/                              # Training, validation, test sets
│   ├── train/
│   ├── valid/
│   └── test/
│
├── notebooks/                         # Jupyter notebooks for EDA, preprocessing, training
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_preprocessing_pipeline.ipynb
│   ├── 04_model_training_resnet50.ipynb
│   ├── 05_model_training_MobileNetV2.ipynb
│   └── 06_model_training_DenseNet121.ipynb
│
├── models/                            # Trained models and accuracy plots
│   ├── DenseNet121/
│   │   └── model_DenseNet121/
│   │       └── best_model_densenet121.h5
│   ├── MobileNetV2/
│   │   └── model_MobileNetV2/
│   │       └── best_model_mobilenetv2.h5
│   └── resnet50/
│       └── model_resnet50/
│           └── best_model_resnet50.h5
│
├── models_evaluation/                 # Evaluation results
│   ├── Evaluation.ipynb
│   └── Evaluation_report/
│       ├── classification_report_*.png
│       └── confusion_matrix_*.png
│
├── models_comparison/                # Accuracy comparison
│   ├── comparison.ipynb
│   └── model_comparison_metrics.png
│
├── data_exploration/                 # Brightness & distribution visuals
│   └── *.png
│
├── Diseases_detection_app/           # Streamlit deployment
│   ├── app.py                        # Streamlit app code
│   ├── best_model_densenet121.h5    # Final model for inference
│   └── classes.json                  # Class label mapping
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview and guide
└── .gitignore                        # Files/folders ignored by Git


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


## 👩‍💻 About Me

- 🙋‍♀️ **Name**: Jyoti  
- 🎓 **Degree**: B.Tech (Artificial Intelligence), 3rd Year  
- 📫 **Email**: [chandilajiya81@gmail.com](mailto:chandilajiya81@gmail.com)  
- 🔗 **LinkedIn**: [www.linkedin.com/in/jyotichandila](https://www.linkedin.com/in/jyotichandila)  
- 💡 **Passion**: Solving real-world problems in agriculture and healthcare using AI  

⭐ *This project reflects my strong interest in combining AI with agriculture to solve impactful, real-world problems. I am actively looking for opportunities to apply my skills in internships and research projects.*



 