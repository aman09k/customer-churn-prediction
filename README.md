# Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer 
will churn or not based on their usage patterns and demographics.

## Problem Statement
Telecom companies lose millions every year due to customer churn. 
This project predicts which customers are likely to leave so the 
company can take action before they churn.

## Key Findings
- Month-to-month customers churn at 43% vs only 3% for two-year contracts
- Most churn happens in the first 10 months
- Churned customers pay $75/month vs $61 for retained customers
- Top predictors: TotalCharges, MonthlyCharges, and Tenure

## Models Used
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 81.55% |
| Random Forest | 79.56% |

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Dataset
- Source: Telco Customer Churn - Kaggle
- 7043 customers, 21 features
- 26.5% churn rate

## How to Run
1. Clone the repository
2. Install requirements:
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
3. Run the app:
streamlit run app.py
