# heart-disease-ml
Machine learning project for heart disease prediction using stacking ensemble and model comparison.
# ❤️ Heart Disease Prediction using Stacking Ensemble

## 📌 Overview

This project focuses on predicting the likelihood of heart disease using machine learning techniques. It implements multiple classification models and improves performance using a **stacking ensemble approach**.

---

## 🎯 Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early prediction using clinical data can help in timely diagnosis and treatment.

---

## 📊 Dataset

* Publicly available heart disease dataset
* Contains medical attributes such as:

  * Age
  * Sex
  * Blood pressure
  * Cholesterol
  * Maximum heart rate
  * Other clinical parameters


---

## ⚙️ Project Workflow

### 1. Data Preprocessing

* Handling missing values
* Feature scaling using StandardScaler
* Train-test split

### 2. Exploratory Data Analysis (EDA)

* Correlation heatmap
* Feature relationships
* Target distribution

### 3. Model Building

Implemented multiple models:

* Logistic Regression
* Random Forest
* Gradient Boosting

### 4. Stacking Ensemble (Key Highlight)

* Combines predictions of base models
* Uses Logistic Regression as meta-learner
* Improves predictive performance

---

## 📈 Evaluation Metrics

* Accuracy
* ROC-AUC Score
* Classification Report

---

## 📊 Results

The stacking model outperformed individual models in overall performance.


---

## 🗂 Project Structure

```
heart-disease-ml/
│
├── data/
├── notebooks/
├── app/
├── models/
├── outputs/
├── requirements.txt
├── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone repository

```
git clone https://github.com/your-username/heart-disease-ml.git
cd heart-disease-ml
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run notebook

Open:

```
notebooks/heart_disease_stacking.ipynb
```

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Model deployment on cloud
* Integration with real-time healthcare systems
* Adding explainable AI (SHAP, LIME)

---

## 🧾 Disclaimer

This project is built using publicly available datasets and is intended for educational and research purposes only.

---

