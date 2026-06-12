import streamlit as st
import pandas as pd
import pickle
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="🔒",
    layout="centered"
)
model = pickle.load(open("model.pkl", "rb"))
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.6rem;
    font-size: 1rem;
    font-weight: 600;
    border: none;
    width: 100%;
}
.stButton > button:hover {
    background-color: #1d4ed8;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
st.title("🔒 UPI Fraud Detection System")
st.markdown(
    "Detect potentially fraudulent financial transactions using Machine Learning."
)
st.divider()
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This application predicts whether a transaction is likely "
        "to be fraudulent using a Random Forest model trained on the "
        "PaySim Synthetic Financial Dataset."
    )
    st.divider()
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precision", "95%")
        st.metric("F1 Score", "97%")
    with col2:
        st.metric("Recall", "99%")
    st.divider()
    st.subheader("📁 Dataset")
    st.write("PaySim Synthetic Dataset")
    st.write("6.3M+ Transactions")
# Input Form
st.subheader("💳 Transaction Details")
transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)
amount = st.number_input(
    "Transaction Amount (₹)",
    min_value=0.0,
    value=0.0
)
st.subheader("👤 Sender Account")
col1, col2 = st.columns(2)
with col1:
    oldbalanceOrg = st.number_input(
        "Balance Before Transaction (₹)",
        min_value=0.0,
        value=0.0
    )
with col2:
    newbalanceOrig = st.number_input(
        "Balance After Transaction (₹)",
        min_value=0.0,
        value=0.0
    )
st.subheader("🏦 Receiver Account")
col3, col4 = st.columns(2)
with col3:
    oldbalanceDest = st.number_input(
        "Receiver Balance Before (₹)",
        min_value=0.0,
        value=0.0
    )
with col4:
    newbalanceDest = st.number_input(
        "Receiver Balance After (₹)",
        min_value=0.0,
        value=0.0
    )
st.divider()
# Prediction
if st.button("🔍 Analyse Transaction"):
    input_data = {
        'step': 1,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_CASH_IN': transaction_type == 'CASH_IN',
        'type_CASH_OUT': transaction_type == 'CASH_OUT',
        'type_DEBIT': transaction_type == 'DEBIT',
        'type_PAYMENT': transaction_type == 'PAYMENT',
        'type_TRANSFER': transaction_type == 'TRANSFER'
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    fraud_probability = round(probability * 100, 2)
    st.subheader("📋 Analysis Result")
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
        st.markdown(
            f"### Fraud Probability: **{fraud_probability}%**"
        )
        st.progress(min(int(fraud_probability), 100))
        st.warning(
            "This transaction shows patterns commonly associated with fraud. "
            "Please verify before proceeding."
        )
    else:
        st.success("✅ Transaction Appears Legitimate")
        st.markdown(
            f"### Fraud Probability: **{fraud_probability}%**"
        )
        st.progress(min(int(fraud_probability), 100))
        st.info(
            "No suspicious patterns were detected in this transaction."
        )