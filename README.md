# Network Intrusion Detection using AI (UNSW-NB15) 🔐

This project implements a Machine Learning based **Network Intrusion Detection System** using the **UNSW-NB15 dataset**.
The system uses **Random Forest Classifier** to detect malicious and normal network traffic and visualizes important insights using graphs and feature analysis.

The project includes preprocessing, feature selection, noise handling, and performance evaluation to build an efficient cybersecurity AI model.

---

## 📌 Objective

To analyze network traffic data and detect cyber attacks using Machine Learning:

* Classify traffic as normal or attack
* Identify most important network features
* Reduce noise and improve model accuracy
* Compare accuracy with different number of features
* Visualize results using graphs and confusion matrix

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Pandas** – Data handling
* **NumPy** – Numerical operations
* **Matplotlib** – Graph visualization
* **Scikit-learn** – Machine Learning
* **Random Forest Classifier**
* **Pipeline / ColumnTransformer** – Preprocessing
* **SimpleImputer / OneHotEncoder** – Data cleaning

---

## 📊 Key Analyses

### 1. Correlation Heatmap

Shows the relationship between numeric features and label to understand important variables.

### 2. Feature Importance (Random Forest)

Top features affecting intrusion detection are selected using Random Forest importance.

### 3. Accuracy vs Number of Features

Model accuracy is compared using different numbers of selected features.

### 4. Top 10 Feature Model

Final optimized model is trained using only the top 10 most important features.

### 5. Noise Handling

Outliers are removed using IQR clipping and low-variance features are removed.


## 📂 Dataset

Dataset used: UNSW-NB15
Source: https://www.kaggle.com/datasets/dhoogla/unswnb15

Dataset is not uploaded due to large size.
Download it from Kaggle.


## 📈 Model Details

* Algorithm: Random Forest
* Train/Test split: 80 / 20
* Feature selection: Top 10 features
* Noise handling: IQR clipping + VarianceThreshold
* Evaluation:

  * Accuracy
  * Classification Report
  * Confusion Matrix
