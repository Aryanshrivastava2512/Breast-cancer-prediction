#  Breast Cancer Classification using Neural Network

##  Project Overview

The **Breast Cancer Classification** project focuses on building a Deep Learning model to predict whether a tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on medical diagnostic features.

Early detection of breast cancer is critical for effective treatment. This project demonstrates how Artificial Neural Networks (ANN) can be applied to solve real-world healthcare problems using machine learning techniques.

The project covers the complete data science pipeline including data preprocessing, exploratory data analysis (EDA), model building, and evaluation.

---

## Problem Statement

The objective of this project is to build a reliable classification model that can:

- Predict whether a tumor is Malignant or Benign  
- Improve diagnostic accuracy  
- Assist in early detection  
- Evaluate model performance using classification metrics  

---

##  Dataset Description

The dataset contains features computed from digitized images of breast mass cell nuclei.

###  Key Features:

- **Radius** – Mean distance from center to perimeter  
- **Texture** – Standard deviation of gray-scale values  
- **Perimeter** – Tumor perimeter  
- **Area** – Tumor area  
- **Smoothness** – Variation in radius lengths  
- **Compactness** – Perimeter² / Area − 1.0  
- **Concavity** – Severity of concave portions  
- **Symmetry** – Tumor symmetry  
- **Fractal Dimension** – Tumor boundary complexity  

###  Target Variable:

- **M** → Malignant (1)  
- **B** → Benign (0)

---

##  Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook  

---

## Project Workflow

###  Data Preprocessing
- Handling missing values  
- Encoding categorical labels (M = 1, B = 0)  
- Feature scaling using StandardScaler  
- Train-Test Split  

###  Exploratory Data Analysis (EDA)
- Distribution analysis  
- Correlation heatmap  
- Class balance visualization  
- Feature relationship analysis  

###  Model Building

An **Artificial Neural Network (ANN)** was implemented using Keras.

####  Model Architecture

- Input Layer: 30 neurons  
- Hidden Layer 1: 16 neurons (ReLU)  
- Hidden Layer 2: 8 neurons (ReLU)  
- Output Layer: 1 neuron (Sigmoid)  

---

###  Model Training

- Optimizer: Adam  
- Loss Function: Binary Crossentropy  
- Evaluation Metric: Accuracy  

---

###  Model Evaluation

Model performance was evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Precision  
- Recall  
- F1 Score  
- Classification Report  

---

## Results

- Achieved high classification accuracy  
- Successfully distinguishes between Malignant and Benign tumors  
- Strong precision and recall values  
- Demonstrates effectiveness of Neural Networks in healthcare prediction  

---

##  How to Run the Project

1. Clone the repository:
git clone https://github.com/Aryanshrivastava2512/Breast-Cancer-Classification.git

---

## Key Insights

- Feature scaling significantly improves neural network performance  
- Certain features strongly correlate with malignancy  
- ANN provides strong classification performance  

---

##  Future Improvements

- Hyperparameter tuning  
- Implement Deep Neural Networks (DNN)  
- Compare with Random Forest, SVM, XGBoost  
- Deploy model using Streamlit or Flask  
- Build an interactive web application  

---

##  Author

**Aryan Shrivastava**  
B.Tech Student | Aspiring Data Scientist  

---

 If you found this project helpful, feel free to give it a star!
