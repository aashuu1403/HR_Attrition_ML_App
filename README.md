# 🧑‍💼 HR Workforce & Attrition Predictive Analytics

## 📌 Project Overview
An end-to-end machine learning pipeline and interactive web application designed to predict employee attrition. This project identifies core flight risks using the IBM HR Analytics dataset and provides HR stakeholders with a real-time decision-support dashboard to evaluate employee retention probabilities.

---

## 🛠️ Tech Stack & Architecture
* **Python (Pandas, Scikit-Learn):** Data cleaning, feature engineering, and predictive modeling.
* **Imbalanced-Learn (SMOTE):** Synthetic minority over-sampling to handle heavily imbalanced human behavior data.
* **Streamlit:** Front-end web framework for deploying the interactive machine learning model.
* **Git/GitHub:** Version control and repository management.

---

## 🧠 Machine Learning Workflow

### 1. Data Processing & Feature Engineering
* Processed 1,470 employee records, converting categorical text variables into numeric boolean columns via One-Hot Encoding.
* Dropped non-predictive baseline features (e.g., `StandardHours`, `EmployeeNumber`) to reduce dimensionality.

### 2. Handling Class Imbalance with SMOTE
* The raw dataset exhibited a severe 84/16 class imbalance (84% retention, 16% attrition).
* Applied Synthetic Minority Over-sampling Technique (SMOTE) strictly to the training data, increasing the minority class recall from a baseline of 0.09 to 0.21, significantly improving the model's ability to catch actual flight risks.

### 3. Model Training & Evaluation
* Trained a **Random Forest Classifier** (`class_weight='balanced'`) on the engineered dataset.
* Extracted feature importances to identify the top drivers of turnover:
  1. Stock Option Level
  2. Monthly Income
  3. Marital Status
  4. Job Satisfaction
  5. Years With Current Manager

---

## 🚀 App Deployment
The trained model (`hr_model.pkl`) is deployed via a Streamlit web application. Users can adjust the top 5 retention drivers using sidebar sliders to calculate real-time churn probability.

### How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/aashuu1403/HR_Attrition_ML_App.git](https://github.com/aashuu1403/HR_Attrition_ML_App.git)