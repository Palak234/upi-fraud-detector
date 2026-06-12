# 🔒 UPI Fraud Detection System

An end-to-end Machine Learning-based fraud detection system that identifies suspicious UPI transactions using transaction patterns and account balance behavior. The project compares multiple classification models and deploys the best-performing model using an interactive Streamlit web application.

---

## 📌 Overview

With the rapid growth of digital payments in India, detecting fraudulent transactions in real time has become a critical challenge.

This project uses supervised machine learning techniques to classify transactions as:

- ✅ Legitimate
- ⚠️ Fraudulent

The model is trained on the PaySim Synthetic Financial Dataset, which simulates real-world mobile money transactions with realistic fraud behavior patterns.

> ⚠️ Note: PaySim is a synthetic dataset. Results may vary on real-world transaction data.

---

## 🚀 Key Features

- End-to-end ML pipeline (data → training → deployment)
- Smart sampling strategy — retained all fraud cases for better model learning
- Data preprocessing and feature engineering
- Handling class imbalance using SMOTE
- Comparison of multiple ML models with detailed evaluation
- Best model selection based on F1 Score and Recall
- Real-time fraud prediction using Streamlit
- Interactive and user-friendly UI
- Model persistence using Pickle

---

## 📊 Dataset

**Dataset:** PaySim Synthetic Financial Dataset — [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total Transactions | 6,362,620 |
| Fraudulent Transactions | 8,213 (0.13%) |
| Normal Transactions Sampled | 100,000 |
| Final Training Dataset Size | 108,213 |

### Input Features

- Transaction Amount
- Transaction Type (One-Hot Encoded)
- Sender Balance Before & After Transaction
- Receiver Balance Before & After Transaction
- Transaction Step (Time Feature)

### Target Variable

| Value | Meaning |
|---|---|
| 0 | Legitimate Transaction |
| 1 | Fraudulent Transaction |

---

## 🛠️ Data Preprocessing

- Removed irrelevant columns: `nameOrig`, `nameDest`, `isFlaggedFraud`
- Applied One-Hot Encoding on transaction type
- Smart sampling: kept all 8,213 fraud cases + 100k normal transactions
- Train-test split (80/20) for model evaluation
- Applied SMOTE to handle class imbalance

### Class Distribution

| | Legitimate | Fraudulent |
|---|---|---|
| Before SMOTE | 80,020 | 6,550 |
| After SMOTE | 80,020 | 80,020 |

---

## 🤖 Machine Learning Models

### 1. Logistic Regression (Baseline)
- Precision: 66% — Recall: 92% — F1: 77%
- Good recall but generates false positives

### 2. Random Forest ⭐ (Final Model)
- Precision: 95% — Recall: 99% — F1: 97%
- Best balance between precision and recall

### 3. XGBoost
- Precision: 96% — Recall: 99% — F1: 98%
- Slightly higher precision but computationally heavier

---

## 🏆 Final Model Selection

**Random Forest Classifier** was chosen for deployment because:

- High precision — minimizes false fraud alerts
- High recall — catches 99% of actual fraud cases
- Fast inference suitable for real-time prediction
- Stable performance on imbalanced dataset with SMOTE

---

## 💻 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Deployment | Streamlit |
| Model Persistence | Pickle |

---

## 📂 Project Structure

```
upi-fraud-detector/
│
├── data/
│   └── transactions.csv
│
├── app.py
├── model.pkl
├── requirements.txt
└── train_model.py
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## ▶️ Model Training

```bash
python train_model.py
```

This will:
- Load and explore full dataset (6.3M rows)
- Apply smart sampling strategy
- Preprocess data and apply SMOTE
- Train and compare 3 ML models
- Save best model as `model.pkl`

---

## 🌐 Run Streamlit App

```bash
streamlit run app.py
```

Enter transaction details to receive:
- Fraud prediction (Legitimate / Fraudulent)
- Fraud probability score
- Visual probability indicator

---

## 🔍 Sample Prediction

**Input:**

| Field | Value |
|---|---|
| Transaction Type | CASH_OUT |
| Amount | ₹5,00,000 |
| Sender Balance Before | ₹5,00,000 |
| Sender Balance After | ₹0 |
| Receiver Balance Before | ₹0 |
| Receiver Balance After | ₹0 |

**Output:**
```
⚠️ Fraudulent Transaction Detected
Fraud Probability: 55%
```

---

## 📈 Future Improvements

- Hyperparameter tuning for better optimization
- Explainable AI using SHAP / LIME
- Real-time transaction API using FastAPI
- Cloud deployment on AWS / Azure
