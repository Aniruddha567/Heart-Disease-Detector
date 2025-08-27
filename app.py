import streamlit as st
import pandas as pd
import joblib
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_or_train_model():
    try:
        if os.path.exists('heart_rf_model.pkl') and os.path.exists('heart_scaler.pkl'):
            model = joblib.load('heart_rf_model.pkl')
            scaler = joblib.load('heart_scaler.pkl')
            st.sidebar.success("✅ Loaded pre-trained model")
            return model, scaler, True
    except Exception as e:
        st.sidebar.warning(f"Could not load model: {str(e)}")
    
    try:
        with st.sidebar:
            with st.spinner("Training new model... This may take a minute"):
                url = "https://raw.githubusercontent.com/Aniruddha567/Heart-Disease-Detector/main/heart_disease_uci.csv"
                df = pd.read_csv(url)
                
                numeric_cols = df.select_dtypes(include='number').columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                
                cat_cols = df.select_dtypes(include='object').columns.tolist()
                if 'num' in cat_cols:
                    cat_cols.remove('num')

                X = df.drop('num', axis=1)
                y = (df['num'] > 0).astype(int)
                
                X = pd.get_dummies(X, columns=cat_cols)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    max_depth=10
                )
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                joblib.dump(model, 'heart_rf_model.pkl')
                joblib.dump(scaler, 'heart_scaler.pkl')
                
                st.sidebar.success(f"✅ Trained new model (Accuracy: {accuracy:.2%})")
                return model, scaler, False
                
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.stop()

def validate_inputs(age, trestbps, chol, thalach, oldpeak):
    errors = []
    if not (18 <= age <= 100):
        errors.append("Age must be between 18-100")
    if not (80 <= trestbps <= 200):
        errors.append("Resting BP must be between 80-200")
    if not (100 <= chol <= 600):
        errors.append("Cholesterol must be between 100-600")
    if not (60 <= thalach <= 220):
        errors.append("Max heart rate must be between 60-220")
    if not (0.0 <= oldpeak <= 6.0):
        errors.append("ST depression must be between 0.0-6.0")
    return errors

model, scaler, model_loaded = load_or_train_model()

st.title("❤️ Heart Disease Detection System")
st.markdown("Predict heart disease risk based on clinical parameters")

risk_threshold = st.sidebar.slider(
    "Risk Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Information")
    
    with st.form("patient_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
                         options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", 
                              options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                           options=["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", 
                        options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", 
                          options=["Normal", "Fixed Defect", "Reversible Defect"])
        
        submit_button = st.form_submit_button("Predict Risk")

with col2:
    st.header("Prediction Results")
    
    if submit_button:
        validation_errors = validate_inputs(age, trestbps, chol, thalach, oldpeak)
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            with st.spinner("Analyzing patient data..."):
                time.sleep(1)
                
                input_data = {
                    'age': age,
                    'trestbps': trestbps,
                    'chol': chol,
                    'thalach': thalach,
                    'oldpeak': oldpeak,
                    'sex_Male': 1 if sex == "Male" else 0,
                    'sex_Female': 1 if sex == "Female" else 0,
                    'cp_Typical Angina': 1 if cp == "Typical Angina" else 0,
                    'cp_Atypical Angina': 1 if cp == "Atypical Angina" else 0,
                    'cp_Non-anginal Pain': 1 if cp == "Non-anginal Pain" else 0,
                    'cp_Asymptomatic': 1 if cp == "Asymptomatic" else 0,
                    'fbs_No': 1 if fbs == "No" else 0,
                    'fbs_Yes': 1 if fbs == "Yes" else 0,
                    'restecg_Normal': 1 if restecg == "Normal" else 0,
                    'restecg_ST-T Wave Abnormality': 1 if restecg == "ST-T Wave Abnormality" else 0,
                    'restecg_Left Ventricular Hypertrophy': 1 if restecg == "Left Ventricular Hypertrophy" else 0,
                    'exang_No': 1 if exang == "No" else 0,
                    'exang_Yes': 1 if exang == "Yes" else 0,
                    'slope_Upsloping': 1 if slope == "Upsloping" else 0,
                    'slope_Flat': 1 if slope == "Flat" else 0,
                    'slope_Downsloping': 1 if slope == "Downsloping" else 0,
                    'ca_0': 1 if ca == 0 else 0,
                    'ca_1': 1 if ca == 1 else 0,
                    'ca_2': 1 if ca == 2 else 0,
                    'ca_3': 1 if ca == 3 else 0,
                    'thal_Normal': 1 if thal == "Normal" else 0,
                    'thal_Fixed Defect': 1 if thal == "Fixed Defect" else 0,
                    'thal_Reversible Defect': 1 if thal == "Reversible Defect" else 0
                }
                
                input_df = pd.DataFrame([input_data])
                
                expected_columns = model.feature_names_in_
                for col in expected_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                input_df = input_df[expected_columns]
                
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                st.success("Analysis complete!")
                
                if probability >= risk_threshold:
                    st.error(f"🚨 High Risk of Heart Disease")
                    st.metric("Probability", f"{probability:.2%}")
                    st.warning("Please consult a cardiologist for further evaluation.")
                else:
                    st.success(f"✅ Low Risk of Heart Disease")
                    st.metric("Probability", f"{probability:.2%}")
                    st.info("Maintain healthy lifestyle with regular checkups.")
                
                st.subheader("Risk Factors Analysis")
                risk_factors = []
                if age > 50:
                    risk_factors.append("⚠️ Age above 50")
                if chol > 240:
                    risk_factors.append("⚠️ High cholesterol level")
                if trestbps > 140:
                    risk_factors.append("⚠️ Elevated blood pressure")
                if exang == "Yes":
                    risk_factors.append("⚠️ Exercise-induced angina present")
                if oldpeak > 2.0:
                    risk_factors.append("⚠️ Significant ST depression")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("✅ No significant risk factors identified")

with st.sidebar:
    st.header("About This App")
    st.info("This app predicts heart disease risk using machine learning.")
    
    st.header("How to Use")
    st.write("1. Fill in patient clinical parameters\n2. Click 'Predict Risk'\n3. Review results and recommendations")
    
    if st.button("Load Sample Patient"):
        st.session_state.age = 55
        st.session_state.trestbps = 140
        st.session_state.chol = 260
        st.session_state.thalach = 150
        st.session_state.oldpeak = 1.2
    
    st.header("Disclaimer")
    st.warning("This tool is for screening purposes only. Always consult healthcare professionals for medical diagnosis.")

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
    }
    @media (max-width: 768px) {
        .stColumn {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)