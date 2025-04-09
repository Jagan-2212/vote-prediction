import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load trained models
rf_best = joblib.load("rf_best.pkl")
xgb_best = joblib.load("xgb_best.pkl")
meta_model = joblib.load("meta_model.pkl")

# Page title
st.title("🗳️ Election Vote Prediction using Stacking Ensemble")

# File upload
uploaded_file = st.file_uploader("Upload CSV file with input data:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview:")
    st.dataframe(df.head())

    # Preprocessing
    try:
        # Handle missing values
        df.fillna({
            "st_name": df["st_name"].mode()[0],
            "pc_type": df["pc_type"].mode()[0],
            "cand_sex": df["cand_sex"].mode()[0],
            "totvotpoll": df["totvotpoll"].mean() if "totvotpoll" in df.columns else 0,
            "electors": df["electors"].mean(),
        }, inplace=True)

        # Feature engineering
        df["voter_turnout"] = (df["totvotpoll"] / df["electors"]) * 100 if "totvotpoll" in df.columns else 0

        # Drop unnecessary columns
        for col in ["cand_name", "partyabbre", "totvotpoll"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Encode categorical variables
        le = LabelEncoder()
        for col in ["st_name", "pc_name", "pc_type", "partyname", "cand_sex"]:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Predict using base models
        rf_preds = rf_best.predict(X_scaled)
        xgb_preds = xgb_best.predict(X_scaled)

        # Stack predictions
        meta_input = np.column_stack((rf_preds, xgb_preds))
        final_preds = meta_model.predict(meta_input)

        # Show predictions
        df["🗳️ Predicted Votes"] = final_preds.astype(int)
        st.subheader("📊 Prediction Results:")
        st.dataframe(df)

        # Download option
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions as CSV", csv_download, file_name="predictions.csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
