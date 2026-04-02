# 🏦 CreditWise – Loan Approval Prediction System

This project builds an intelligent loan approval system for financial institutions using supervised machine learning. The goal is to automate decision-making, reduce risk, and ensure consistent, data-driven approvals.

---

## Problem

Traditional loan approval is manual, slow, and inconsistent. This leads to:

- Rejection of good customers → loss of business  
- Approval of high-risk customers → financial losses  

This project solves the problem by predicting loan approval based on applicant financial and demographic data.

---

## Dataset

Each record represents a loan applicant with features including:

- Financial: Income, Savings, Loan Amount, DTI Ratio  
- Credit: Credit Score, Existing Loans  
- Personal: Age, Employment Status, Dependents  
- Loan Info: Loan Purpose, Loan Term, Collateral Value  
- Demographics: Property Area, Education, Gender  

Target:
- `Loan_Approved` → 1 (Approved), 0 (Rejected)

---

## CreditWise ML System

### Approach

Project workflow:

1. Data Cleaning & Missing Value Handling
2. EDA (Exploratory Data Analysis)
3. Feature Encoding (One-Hot Encoding)  
4. Feature Scaling (Standardization)  
5. Model Building using Pipeline  
6. Hyperparameter Tuning (GridSearchCV)  
7. Model Evaluation (Precision, Recall, Accuracy)  
8. Business-focused optimization  

---

## Models

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  

### Insights

- Logistic Regression & Naive Bayes → Balanced performance  
- KNN → Higher precision (useful for risk-sensitive decisions)  

---

## Evaluation Strategy

The system is designed with a **risk-aware objective**:

- Precision prioritized → minimizes approval of risky customers  
- Recall monitored → avoids rejecting good customers  

This reflects a real-world **cost-sensitive decision framework**.

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## Key Highlights

- End-to-end ML pipeline using scikit-learn  
- Integrated preprocessing + model training using Pipeline  
- Hyperparameter tuning with GridSearchCV  
- Focus on business-relevant metrics over accuracy  

---

## Future Scope

- Deploy as a web application (Flask / Streamlit)  
- Add explainability (SHAP / LIME)  
- Improve recall using ensemble models  
- Integrate real-time loan scoring API  

---

## Author

**Deepanshu Tevathiya**
AI / ML Enthusiast
