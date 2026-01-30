import streamlit as st
import pandas as pd
import joblib

# --- Load model and columns ---
model = joblib.load('LogisticRegression.pkl')
columns = joblib.load('columns.pkl')  # 15 columns

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below:")

# --- Human-readable inputs ---
age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
gender = st.selectbox("Gender", ["Male", "Female"])
fastingbs = st.selectbox("High Fasting Blood Sugar?", ["No", "Yes"])
exercise_angina = st.selectbox("Exercise Angina?", ["No", "Yes"])
oldpeak = st.number_input("ST Depression (Oldpeak)", value=0.0)

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP"])  # only the two in your model
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST"])
st_slope = st.selectbox("ST Slope", ["Flat", "Up"])
bp = st.selectbox("BP Category", ["High Stage2", "Other"])  # only High Stage2 in your model
chol = st.selectbox("Cholesterol", ["Borderline", "Other"])
maxhr = st.selectbox("Max Heart Rate", ["Below Avg", "High"])

# --- Convert to model input ---
input_data = {
    'Age': age / 100,  # simple scaling, adjust if you know the exact scaling
    'isfemale': 1 if gender=="Female" else 0,
    'FastingBS': 1 if fastingbs=="Yes" else 0,
    'isExerciseAngina': 1 if exercise_angina=="Yes" else 0,
    'Oldpeak': oldpeak,  # use exact scaling if known
    'ChestPainType_ATA': 1 if chest_pain=="ATA" else 0,
    'ChestPainType_NAP': 1 if chest_pain=="NAP" else 0,
    'RestingECG_Normal': 1 if rest_ecg=="Normal" else 0,
    'RestingECG_ST': 1 if rest_ecg=="ST" else 0,
    'ST_Slope_Flat': 1 if st_slope=="Flat" else 0,
    'ST_Slope_Up': 1 if st_slope=="Up" else 0,
    'RestingBP_cat_High_Stage2': 1 if bp=="High Stage2" else 0,
    'Cholesterol_cat_Borderline': 1 if chol=="Borderline" else 0,
    'MaxHR_cat_Below_Avg': 1 if maxhr=="Below Avg" else 0,
    'MaxHR_cat_High': 1 if maxhr=="High" else 0
}

df_input = pd.DataFrame([input_data])
df_input = df_input.reindex(columns=columns, fill_value=0)

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\nProbability: {probability:.2f}")
