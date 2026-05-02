# 💳 Payment Fraud Detection using ML

![Python](https://img.shields.io/badge/Python-3.9+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-blue.svg)

**Detect fraudulent online payment transactions in real-time using Machine Learning with explainable predictions and risk scoring.**

---

## 🎯 Overview

This is a **Machine Learning based Payment Fraud Detection System** built with Streamlit. It allows users to predict whether a transaction is fraudulent or not — either one at a time or in bulk via CSV upload. The system provides a risk score, a plain-English explanation for every prediction, and sends an email alert when fraud is detected.

### ✨ Key Highlights

- **🔍 Single Transaction Prediction** — Enter transaction details manually and get instant fraud/not-fraud result
- **🏦 Bulk CSV Upload** — Upload a bank transaction file and predict all transactions at once
- **📊 Interactive Dashboard** — Visualize fraud distribution, risk scores, and transaction patterns
- **🧠 Explainable Predictions** — Every result comes with a human-readable reason
- **📧 Email Alerts** — Automatic email notification when a fraudulent transaction is detected
- **⚡ Risk Scoring** — Each transaction gets a risk score from 0% to 100%

---

## 🏗️ Project Structure

```
Fraud codes/
├── app/
│   ├── fraud_detection.py      ← Main polished Streamlit app
│   ├── app.py                  ← Extended app version
│   └── 1.py                    ← Basic starter version
│
├── models/
│   ├── fraud_detection_pipeline.pkl   ← Final production model
│   ├── xgboost_model.pkl              ← XGBoost model
│   ├── gradient_boosting_model.pkl    ← Gradient Boosting model
│   ├── logistic_regression_model.pkl  ← Logistic Regression model
│   └── random_forest_model.pkl        ← Random Forest model
│
├── notebooks/
│   ├── analysis_model.ipynb       ← Model training & analysis
│   ├── balance.ipynb              ← Dataset balancing experiments
│   ├── compare_models.ipynb       ← Model comparison & selection
│   └── risk_score_calculator.ipynb ← Risk score logic
│
├── data/                          ← Dataset folder (not pushed to GitHub)
├── requirements.txt
└── README.md
```

---

## 🚀 Features

### Single Transaction Prediction
- Input 6 fields: transaction type, amount, sender/receiver balances
- Get instant **Fraud / Not Fraud** result
- See **Risk Score %** and plain-English **explanation**
- Automatic **email alert** sent if fraud is detected

### Bank Transactions (Bulk)
- Upload a CSV file with multiple transactions
- Predict all transactions at once
- View results with risk scores and reasons in a table

### Dashboard
- **Total Transactions** counter
- **Fraud vs Not Fraud** pie chart
- **Risk Score Distribution** histogram
- **Transaction Type vs Amount** box plot
- Full predictions table

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Streamlit |
| ML Models | Scikit-learn, XGBoost |
| Data Processing | Pandas |
| Visualization | Plotly |
| Model Storage | Joblib (.pkl) |
| Email Alerts | smtplib (Gmail SMTP) |
| Language | Python 3.9+ |

---

## 🤖 ML Models Used

Four models were trained and compared (see `notebooks/compare_models.ipynb`):

| Model | File |
|-------|------|
| Random Forest | `random_forest_model.pkl` |
| XGBoost | `xgboost_model.pkl` |
| Gradient Boosting | `gradient_boosting_model.pkl` |
| Logistic Regression | `logistic_regression_model.pkl` |

The best performing model was saved as `fraud_detection_pipeline.pkl` and used in production.

---

## 📋 Input Features

The model takes these 6 inputs:

| Feature | Description |
|---------|-------------|
| `type` | Transaction type: PAYMENT, TRANSFER, CASH_OUT |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender's balance before transaction |
| `newbalanceOrig` | Sender's balance after transaction |
| `oldbalanceDest` | Receiver's balance before transaction |
| `newbalanceDest` | Receiver's balance after transaction |

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/viswadhar88/Payment-fraud-detection-using-ML.git
cd Payment-fraud-detection-using-ML
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
cd app
streamlit run fraud_detection.py
```

App will open at `http://localhost:8501`

---

## 📊 CSV Upload Format

For bulk prediction, your CSV must have these columns:
```
type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
```

Example:
```csv
type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
PAYMENT,1000,10000,9000,0,0
TRANSFER,200000,500000,300000,0,200000
```

---

## 📧 Email Alert Setup

To enable email alerts, update these lines in `app/fraud_detection.py`:
```python
sender_email = "your_email@gmail.com"
sender_password = "your_app_password"
receiver_email = "alert_receiver@gmail.com"
```

> Use a Gmail App Password, not your regular Gmail password.

---

## 🧪 Notebooks

| Notebook | Purpose |
|----------|---------|
| `analysis_model.ipynb` | Full model training and performance analysis |
| `balance.ipynb` | Dataset balancing techniques (oversampling/undersampling) |
| `compare_models.ipynb` | Side-by-side comparison of all 4 ML models |
| `risk_score_calculator.ipynb` | Risk score calculation logic and testing |

---

## 🐛 Troubleshooting

**Model not loading?**
- Make sure `fraud_detection_pipeline.pkl` is inside the `models/` folder
- The path in code should be `models/fraud_detection_pipeline.pkl`

**Email alert not working?**
- Enable 2-step verification on Gmail
- Generate an App Password from Google Account settings
- Use that App Password in the code

---

## 📞 Contact

**K Venkata Viswadhar**

GitHub: [@viswadhar88](https://github.com/viswadhar88)

Project Link: [https://github.com/viswadhar88/Payment-fraud-detection-using-ML](https://github.com/viswadhar88/Payment-fraud-detection-using-ML)

---

*Built using Python | Streamlit | Machine Learning | Explainable AI*
