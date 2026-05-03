# fraud_detection.py — Polished & Efficient Version (Same Layout, Enhanced UI)
import shap
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# =====================================
# PAGE CONFIGURATION & GLOBAL STYLING
# =====================================
st.set_page_config(page_title="Fraud Detection System", layout="wide", page_icon="💳")

st.markdown("""
<style>
/* General Page Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Navbar Buttons */
div.stButton > button {
    width: 220px; height: 45px; font-size: 16px;
    border-radius: 8px; font-weight: 600;
    background-color: #1e3c72; color: white;
    border: none; box-shadow: 0px 3px 6px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #2a5298; transform: scale(1.05);
}

/* Metric Boxes */
.metrics-container { 
    display: flex; justify-content: space-around; flex-wrap: wrap; 
    margin: 20px 0; 
}
.metric-box {
    flex: 1 1 28%; min-width: 250px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 15px; padding: 25px 15px;
    text-align: center; margin: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}
.metric-title { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.metric-value { font-size: 26px; font-weight: bold; }

.total-transactions .metric-value { color: #00BFFF; }
.fraud-transactions { background-color: rgba(255,99,71,0.85); color: white; }
.not-fraud-transactions { background-color: rgba(144,238,144,0.85); color: black; }

/* Card-like content containers */
.glass-box {
    backdrop-filter: blur(20px);
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 40px;
    color: white;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    text-align: center;
    max-width: 950px;
    margin: 30px auto;
}

/* Animation */
.fade-in { animation: fadeIn 1.5s ease-in-out; }
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Tables */
.dataframe tbody tr:hover {
    background-color: rgba(255,255,255,0.1);
}

/* Section Headings */
h1, h2, h3 {
    text-align: center;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# SESSION STATE INITIALIZATION
# =====================================
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "single_predictions" not in st.session_state:
    st.session_state.single_predictions = pd.DataFrame(
        columns=["Type","Amount","Old Balance (Sender)","New Balance (Sender)",
                 "Old Balance (Receiver)","New Balance (Receiver)",
                 "Result","Risk Score","Reason"])
if "file_predictions" not in st.session_state:
    st.session_state.file_predictions = pd.DataFrame()

# =====================================
# HELPER FUNCTIONS
# =====================================
def set_page(new_page):
    st.session_state.page = new_page

def centered_title(title, icon=None):
    if icon:
        st.markdown(f"<h1 style='text-align:center;'>{icon} {title}</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align:center;'>{title}</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
       import os
       return joblib.load(os.path.join(os.path.dirname(__file__), "../models/fraud_detection_pipeline.pkl"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---------- EMAIL ALERT ----------
def send_email_alert(transaction_info):
    sender_email = "luffyisbot@gmail.com"
    sender_password = "your_app_password_here"
    receiver_email = "luffyisbot@gmail.com"

    subject = "🚨 Fraudulent Transaction Detected"
    body = f"""
    ⚠️ Fraudulent Transaction Alert ⚠️

    Type: {transaction_info.get('Type')}
    Amount: {transaction_info.get('Amount')}
    Sender Old Balance: {transaction_info.get('Old Balance (Sender)')}
    Sender New Balance: {transaction_info.get('New Balance (Sender)')}
    Receiver Old Balance: {transaction_info.get('Old Balance (Receiver)')}
    Receiver New Balance: {transaction_info.get('New Balance (Receiver)')}
    Reason: {transaction_info.get('Reason')}
    Risk Score: {transaction_info.get('Risk Score')}%

    Please verify this transaction immediately.
    """

    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = sender_email, receiver_email, subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.toast("📧 Fraud alert email sent successfully.")
    except Exception as e:
        st.warning(f"Email alert failed: {e}")

# ---------- Prediction Logic ----------
def explain_prediction(model, df_input):
    try:
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        preprocessed = model.named_steps['preprocessor'].transform(df_input)
        shap_values = explainer.shap_values(preprocessed)
        
        feature_names = ["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
        
        # For fraud class (index 1)
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]
        
        top_features = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        reasons = [f"{f} (impact: {v:+.3f})" for f, v in top_features]
        return " | ".join(reasons)
    except:
        return "Explanation unavailable"

def predict_transaction(model, t_type, amount, old_org, new_org, old_dest, new_dest):
    balance_diff = old_org - new_org
    error_balance = abs(amount - balance_diff)
    is_empty_dest = 1 if old_dest == 0 else 0

    df = pd.DataFrame([{
        "type": t_type, "amount": amount,
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "balance_diff": balance_diff,
        "error_balance": error_balance,
        "is_empty_dest": is_empty_dest
    }])
    proba = model.predict_proba(df)[0][1]
    threshold = st.session_state.get("threshold", 0.5)
    pred = "Fraud" if proba > threshold else "Not Fraud"
    risk_score = int(round(proba * 100))
    reason = explain_prediction(model, df)
    return pred, risk_score, reason

# =====================================
# NAVIGATION BAR
# =====================================
nav_labels = ["🏠 Home","🔍 Single Prediction","🏦 Bank Transactions","📊 Dashboard","📈 Model Performance","👥 About Us"]
nav_pages = ["Home","Single Prediction","Bank Transactions","Dashboard","Model Performance","About Us"]
with st.container():
    cols = st.columns(len(nav_labels))
    for col, label, page in zip(cols, nav_labels, nav_pages):
        if col.button(label):
            set_page(page)
st.markdown("---")

# =====================================
# PAGE: HOME
# =====================================
if st.session_state.page == "Home":
    st.markdown("""
    <div class="glass-box fade-in">
        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="120" style="margin-bottom:20px;">
        <h1>Online Payment Fraud Detection System</h1>
        <p style="font-size:18px; line-height:1.6;">
            Detect, explain, and visualize fraudulent transactions in real-time with intelligent ML-backed insights.
        </p>
        <ul style="text-align:left; font-size:17px; margin:20px auto; display:inline-block;">
            <li>Real-time prediction with detailed reasoning</li>
            <li>Bank transaction analysis with risk scores</li>
            <li>Interactive visualization dashboard</li>
            <li>Automated fraud alert email notifications</li>
        </ul>
        <p><em>Built using Streamlit | Machine Learning | Explainable AI</em></p>
    </div>
    """, unsafe_allow_html=True)

# =====================================
# PAGE: SINGLE PREDICTION
# =====================================
elif st.session_state.page == "Single Prediction":
    centered_title("Single Transaction Prediction", "🔍")
    st.session_state["threshold"] = st.slider("🎚️ Detection Threshold", 0.1, 0.9, 0.5, 0.05,
    help="Lower = catch more frauds but more false alarms. Higher = fewer alerts but may miss frauds.")
    model = load_model()
    if model:
        with st.form("single_pred_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                t_type = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER","CASH_OUT"])
            with col2:
                amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
            with col3:
                old_org = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
            col4, col5, col6 = st.columns(3)
            with col4:
                new_org = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
            with col5:
                old_dest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
            with col6:
                new_dest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

            submitted = st.form_submit_button("Predict Transaction 💡")
        
        if submitted:
            result, risk_score, reason = predict_transaction(model, t_type, amount, old_org, new_org, old_dest, new_dest)
            if result == "Fraud":
                st.error(f"🚨 Prediction: {result} | Risk Score: {risk_score}%")
            else:
                st.success(f"✅ Prediction: {result} | Risk Score: {risk_score}%")
            st.info(f"🧠 Explanation: {reason}")

            if result == "Fraud":
                send_email_alert({
                    "Type": t_type, "Amount": amount,
                    "Old Balance (Sender)": old_org, "New Balance (Sender)": new_org,
                    "Old Balance (Receiver)": old_dest, "New Balance (Receiver)": new_dest,
                    "Result": result, "Risk Score": risk_score, "Reason": reason
                })

            new_row = pd.DataFrame([{
                "Type": t_type, "Amount": amount,
                "Old Balance (Sender)": old_org, "New Balance (Sender)": new_org,
                "Old Balance (Receiver)": old_dest, "New Balance (Receiver)": new_dest,
                "Result": result, "Risk Score": risk_score, "Reason": reason
            }])
            st.session_state.single_predictions = pd.concat(
                [st.session_state.single_predictions, new_row], ignore_index=True
            )

# =====================================
# PAGE: BANK TRANSACTIONS
# =====================================
elif st.session_state.page == "Bank Transactions":
    centered_title("Bank Transactions Fraud Detection", "🏦")
    st.info("Upload your transaction CSV with columns: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest")
    model = load_model()
    if model:
        uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if st.button("Predict All Transactions ⚙️"):
                df["Result"], df["Risk Score"], df["Reason"] = zip(*df.apply(
                    lambda row: predict_transaction(
                        model, row["type"], row["amount"],
                        row["oldbalanceOrg"], row["newbalanceOrig"],
                        row["oldbalanceDest"], row["newbalanceDest"]
                    ), axis=1
                ))
                st.session_state.file_predictions = pd.concat(
                    [st.session_state.file_predictions, df], ignore_index=True
                )
                st.success("✅ Predictions generated successfully.")
                st.dataframe(df)

# =====================================
# PAGE: DASHBOARD
# =====================================
elif st.session_state.page == "Dashboard":
    centered_title("Fraud Detection Dashboard", "📊")
    combined = pd.concat(
        [st.session_state.single_predictions, st.session_state.file_predictions],
        ignore_index=True
    )
    if not combined.empty:
        st.markdown(f"""
        <div class="metrics-container fade-in">
            <div class="metric-box total-transactions">
                <div class="metric-title">Total Transactions</div>
                <div class="metric-value">{len(combined)}</div>
            </div>
            <div class="metric-box fraud-transactions">
                <div class="metric-title">Fraud Transactions</div>
                <div class="metric-value">{combined["Result"].value_counts().get("Fraud",0)}</div>
            </div>
            <div class="metric-box not-fraud-transactions">
                <div class="metric-title">Not Fraud Transactions</div>
                <div class="metric-value">{combined["Result"].value_counts().get("Not Fraud",0)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(combined, names="Result", title="Fraud vs Not Fraud Distribution",
                          color="Result", color_discrete_map={"Fraud":"lightcoral","Not Fraud":"lightgreen"})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(combined, x="Risk Score", nbins=20, color="Result",
                                title="Fraud Risk Score Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(combined, x="Type", y="Amount", color="Result",
                      title="Transaction Type vs Amount by Result")
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(combined)
    else:
        st.info("No transactions to display yet. Upload or predict to view insights.")
# =====================================
# PAGE: MODEL PERFORMANCE
# =====================================
elif st.session_state.page == "Model Performance":
    centered_title("Model Performance", "📈")

    combined = pd.concat(
        [st.session_state.single_predictions, st.session_state.file_predictions],
        ignore_index=True
    )

    if combined.empty or "Result" not in combined.columns:
        st.info("⚠️ No predictions yet. Go to Single Prediction or Bank Transactions first, then come back here.")
    else:
        # --- Confusion Matrix ---
        st.subheader("🟦 Confusion Matrix")
        y_true = combined["Result"].apply(lambda x: 1 if x == "Fraud" else 0)
        y_pred = combined["Risk Score"].apply(lambda x: 1 if x >= 50 else 0)

        cm = confusion_matrix(y_true, y_pred)
        labels = ["Not Fraud", "Fraud"]

        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True
        )
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # --- ROC Curve ---
        st.subheader("📉 ROC-AUC Curve")
        y_score = combined["Risk Score"] / 100
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f"ROC Curve (AUC = {roc_auc:.2f})",
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            color_discrete_sequence=["#1e3c72"]
        )
        fig_roc.add_shape(
            type="line", line=dict(dash="dash", color="red"),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # --- Classification Report ---
        st.subheader("📊 Classification Report")
        report = classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df)

        # --- Key Metrics ---
        st.subheader("🎯 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC Score", f"{roc_auc:.2f}")
        col2.metric("Precision (Fraud)", f"{report['Fraud']['precision']:.2f}")
        col3.metric("Recall (Fraud)", f"{report['Fraud']['recall']:.2f}")
        col4.metric("F1 Score (Fraud)", f"{report['Fraud']['f1-score']:.2f}")
# =====================================
# PAGE: ABOUT US
# =====================================
elif st.session_state.page == "About Us":
    centered_title("About Us", "👥")
    st.markdown("""
    <div class="glass-box fade-in">
        <img src="https://cdn-icons-png.flaticon.com/512/9131/9131529.png" width="120" style="margin-bottom:15px;">
        <h3>Developed by Charan Kasanneni</h3>
        <p style="font-size:18px;">Designed for real-time fraud detection with transparency, explainability, and AI-powered insights.</p>
        <p><strong>Technologies:</strong> Streamlit | Python | Machine Learning | Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
