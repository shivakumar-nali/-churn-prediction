import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="📉 Churn Prediction",
    layout="wide",
    page_icon="📉"
)

# Load model and columns
model = joblib.load("churn_log_model.pkl")
columns = joblib.load("model_columns.pkl")

# Title
st.title("📉 Customer Churn Prediction App")
st.markdown("""
This smart ML app predicts whether a customer is likely to **churn**  
based on their behavior and demographics.

✅ Logistic Regression Model  
✅ Probability of churn  
✅ View actual label (if using sample)
""")

# Sidebar
st.sidebar.title("🔍 Choose Input Type")
input_mode = st.sidebar.radio("Input Mode", ["Manual Input", "Random Sample"])

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_churn_data.csv")

df = load_data()

# Prepare input
if input_mode == "Manual Input":
    user_input = {}
    st.subheader("🧾 Enter Customer Details")
    cols = st.columns(3)

    for i, col_name in enumerate(columns):
        with cols[i % 3]:
            val = st.number_input(
                col_name,
                step=1.0,
                format="%.2f",
                key=col_name
            )
            user_input[col_name] = val

    input_df = pd.DataFrame([user_input])
    actual_label = None

else:
    st.subheader("🎲 Random Customer Sample")
    random_row = df.sample(1).reset_index(drop=True)
    input_df = random_row[columns]
    st.write("🔍 **Sample Data:**")
    st.dataframe(random_row)

    # Assume 'Churn' column is the target
    actual_label = random_row["Churn"].values[0]

# Predict button
if st.button("🚀 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📢 Prediction Result")
    if prediction == 1:
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **not likely to churn**.")

    st.info(f"📊 **Churn Probability:** `{probability:.2%}`")

    if actual_label is not None:
        label_text = "Likely to Churn" if actual_label == 1 else "Not Likely to Churn"
        st.markdown(f"🧷 **Actual Label from Dataset:** `{label_text}`")

    st.caption("✅ Prediction complete. Use the sidebar to try other inputs.")
