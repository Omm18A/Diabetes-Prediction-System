streamlit_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

model = tf.keras.models.load_model("diabetes_mlp.keras")
preprocessor = joblib.load("diabetes_preprocessor.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient information to obtain a model prediction and probability.")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 1.0, 300.0, 100.0)
blood_pressure = st.number_input("Blood Pressure", 1.0, 200.0, 70.0)
skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
bmi = st.number_input("BMI", 1.0, 80.0, 25.0)
pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.3)
age = st.number_input("Age", 1, 120, 25)

if st.button("Predict Diabetes"):

    data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [pedigree],
        "Age": [age]
    })

    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        data[col] = data[col].replace(0, np.nan)

    data["BMI_Category"] = pd.cut(
        data["BMI"], [0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

    data["Age_Group"] = pd.cut(
        data["Age"], [0, 25, 40, 60, np.inf],
        labels=["Young", "Adult", "Middle_Aged", "Senior"]
    )

    data["Glucose_Category"] = pd.cut(
        data["Glucose"], [0, 100, 125, np.inf],
        labels=["Normal", "Elevated", "High"]
    )

    data["Glucose_BMI_Interaction"] = data["Glucose"] * data["BMI"]

    processed = preprocessor.transform(data)
    probability = float(model.predict(processed, verbose=0)[0][0])

    prediction = "Diabetic" if probability >= 0.5 else "Non-Diabetic"

    st.subheader("Prediction Result")
    st.write(f"### Prediction: {prediction}")
    st.write(f"### Diabetes Probability: {probability * 100:.2f}%")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

print("app.py created successfully.")
print("Run with: streamlit run app.py")
