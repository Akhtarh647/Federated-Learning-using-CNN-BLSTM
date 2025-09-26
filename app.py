import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# -------------------------
# Attack type mapping and descriptions
# -------------------------
attack_mapping_full = {
    0: "Normal",
    1: "DDoS_ICMP",
    2: "DDoS_HTTP",
    3: "DDoS_TCP",
    4: "DDoS_UDP",
    5: "DoS_ICMP",
    6: "DoS_HTTP",
    7: "DoS_TCP",
    8: "DoS_UDP",
    9: "BruteForce",
    10: "WebAttack",
    11: "Infiltration",
    12: "Bot",
    13: "PortScan",
    14: "MITM"
}

attack_descriptions = {
    "Normal": "Normal network traffic with no malicious activity",
    "DDoS_ICMP": "Distributed ICMP flood from multiple sources",
    "DDoS_HTTP": "Distributed HTTP flood attack from multiple sources",
    "DDoS_TCP": "Distributed TCP SYN flood attack from multiple sources",
    "DDoS_UDP": "Distributed UDP flood attack from multiple sources",
    "DoS_ICMP": "Single-source ICMP flood attack",
    "DoS_HTTP": "Single-source HTTP flood attack",
    "DoS_TCP": "Single-source TCP SYN flood attack",
    "DoS_UDP": "Single-source UDP flood attack",
    "BruteForce": "Password guessing through repeated login attempts",
    "WebAttack": "Exploiting vulnerabilities in web applications",
    "Infiltration": "Unauthorized internal network access",
    "Bot": "Botnet activity - compromised devices controlled remotely",
    "PortScan": "Scanning network for open ports",
    "MITM": "Man-in-the-Middle - intercepting and altering communications"
}

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Federated IDS Predictor", layout="centered")
st.title(" Intrusion Detection System (IDS) Predictor")
st.markdown("""
This app uses a **Hybrid CNN + LSTM** deep learning model to detect and classify cyber attacks from Edge-IIoT network data.
Upload your dataset below to get real-time predictions.
""")

# Load pre-trained model and encoders
@st.cache_resource
def load_model_and_tools():
    model = load_model("hybrid_cnn_lstm_multiclass.h5")
    label_encoder = joblib.load("attack_label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, label_encoder, scaler

model, label_encoder, scaler = load_model_and_tools()

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader(" Upload a CSV file containing network features", type=["csv"])

# -------------------------
# When user uploads a file
# -------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        # Drop irrelevant columns if present
        drop_cols = ['frame.time', 'ip.src_host', 'ip.dst_host', 'Attack_type', 'Attack_label']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Encode object columns
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.factorize(df[col])[0]

        # Scale and reshape
        X_scaled = scaler.transform(df)
        X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Predict
        st.subheader("Making Predictions")
        predictions = model.predict(X_input)
        pred_classes = np.argmax(predictions, axis=1)

        # Map to attack type and description
        pred_attack_names = [attack_mapping_full.get(c, "Unknown") for c in pred_classes]
        pred_attack_desc = [attack_descriptions.get(name, "No description available") for name in pred_attack_names]

        # Show results
        df_results = df.copy()
        df_results['Predicted_Class'] = pred_classes
        df_results['Predicted_Attack_Type'] = pred_attack_names
        df_results['Description'] = pred_attack_desc

        st.success("Prediction Complete!")
        st.dataframe(df_results[['Predicted_Class', 'Predicted_Attack_Type', 'Description']].head(10))

        # Download
        csv = df_results.to_csv(index=False)
        st.download_button("Download Predictions as CSV", csv, "predicted_attacks.csv", "text/csv")

        # Show reference mapping table
        st.subheader("Attack Type Reference")
        ref_df = pd.DataFrame([
            {"Class ID": k, "Attack Type": v, "Description": attack_descriptions[v]} 
            for k, v in attack_mapping_full.items()
        ])
        st.dataframe(ref_df)

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a properly formatted CSV file to begin.")
