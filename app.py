import streamlit as st
import pandas as pd
import joblib
import sqlite3

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("freight_model.pkl")

# -----------------------------
# Database Connection
# -----------------------------
conn = sqlite3.connect("inventory.db", check_same_thread=False)

# -----------------------------
# App Title
# -----------------------------
st.title("📊 Vendor Invoice Intelligence System")
st.write(
    "Enter a new invoice below to predict expected freight "
    "and assess invoice risk."
)

st.divider()

# -----------------------------
# Invoice Input
# -----------------------------
st.subheader("Enter New Invoice")

col1, col2 = st.columns(2)

with col1:
    dollars = st.number_input(
        "Invoice Amount ($)",
        min_value=0.0,
        value=18500.0,
        step=100.0
    )

with col2:
    actual_freight = st.number_input(
        "Actual Freight Cost ($)",
        min_value=0.0,
        value=200.0,
        step=10.0
    )

# -----------------------------
# Analyze Invoice
# -----------------------------
if st.button("Analyze Invoice", type="primary"):

    # Predict expected freight
    expected_freight = model.predict(
        pd.DataFrame({
            "Dollars": [dollars]
        })
    )[0]

    # Calculate deviation
    deviation_percentage = (
        (actual_freight - expected_freight)
        / expected_freight
    ) * 100

    # Determine risk
    if deviation_percentage > 50:
        risk = "HIGH"
    elif deviation_percentage > 20:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.divider()

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("Invoice Analysis")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Expected Freight",
            f"${expected_freight:.2f}"
        )

    with result_col2:
        st.metric(
            "Freight Deviation",
            f"{deviation_percentage:.2f}%"
        )

    with result_col3:
        if risk == "HIGH":
            st.error(f"Risk Level: {risk}")
        elif risk == "MEDIUM":
            st.warning(f"Risk Level: {risk}")
        else:
            st.success(f"Risk Level: {risk}")
