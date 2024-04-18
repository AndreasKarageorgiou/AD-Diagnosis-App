import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def get_clean_data():
    df = pd.read_csv("AIBL.csv")
    # Apply any necessary transformations as during training
    diagnosis_features = ['DXNORM', 'DXMCI', 'DXAD']
    df['DXTYPE'] = df.apply(lambda row: 1 if row['DXMCI'] == 1 else 2 if row['DXAD'] == 1 else 0, axis=1)
    df.drop(diagnosis_features + ['APTyear', 'Examyear', 'PTDOBYear', 'DXCURREN'], axis=1, inplace=True)
    return df

def load_model_and_scaler():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")  # Ensure you have saved and now load the scaler
    return model, scaler

def prepare_input(input_dict):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])
    # Convert all inputs to float
    input_df = input_df.astype(float)
    return input_df

def add_sidebar():
    st.sidebar.header("Patient Input Form")
    data = get_clean_data()
  
    # Convert categorical input to binary/numerical inputs for model compatibility
    gender_options = {"Male": 1, "Female": 2}
    binary_options = {"Yes": 1, "No": 0}
  
    input_dict = {
        "PTGENDER": gender_options[st.sidebar.selectbox("Select patient's gender:", ["Male", "Female"])],
        "ExamAge": st.sidebar.number_input("Enter patient's age:", min_value=50, max_value=110),
        "MH16SMOK": binary_options[st.sidebar.radio("Is / Was the Patient Smoking?", ["Yes", "No"])],
        "MH2NEURL": binary_options[st.sidebar.radio("Neurological Assessment (Normal/Abnormal):", ["Yes", "No"])],
        "MH8MUSCL": binary_options[st.sidebar.radio("Musculoskeletal Assessment Normal:", ["Yes", "No"])],
        "MHPSYCH": binary_options[st.sidebar.radio("Does the individual have a positive Mental Health Psych Score?", ["Yes", "No"])],
        "MH10GAST": binary_options[st.sidebar.radio("Is the Gastrointestinal Health within normal limits?", ["Yes", "No"])],
        "MH4CARD": binary_options[st.sidebar.radio("Does the individual have an elevated Cardiovascular Risk?", ["Yes", "No"])],
        "MH9ENDO": binary_options[st.sidebar.radio("Is the Endocrine Function within the normal range?", ["Yes", "No"])],
        "MH17MALI": binary_options[st.sidebar.radio("Is the individual at risk for malignancy?", ["Yes", "No"])],
        "MH6HEPAT": binary_options[st.sidebar.radio("Are there any abnormalities in the Liver Function Marker?", ["Yes", "No"])],
        "MH12RENA": binary_options[st.sidebar.radio("Is the Renal Function within normal limits?", ["Yes", "No"])]
    }

    # Slider inputs remain numerical and don't need mapping
    input_dict.update({
        "APGEN1": st.sidebar.slider("Enter APGEN1 value:", 2, 4, 3),
        "APGEN2": st.sidebar.slider("Enter APGEN2 value:", 2, 4, 3),
        "CDGLOBAL": st.sidebar.slider("Cognitive Decline Score (0-5):", 0, 5, 2),
        "AXT117": st.sidebar.slider("Thyroid Stimulating Hormone-Blood Test Result (0-1300):", 0, 1300, 650),
        "BAT126": st.sidebar.slider("Vitamin B12 level (100-2500):", 100, 2500, 1300),
        "HMT3": st.sidebar.slider("RBC Count (5-584):", 5, 584, 290),
        "HMT7": st.sidebar.slider("WBC Count (3-145):", 3, 145, 74),
        "HMT13": st.sidebar.slider("Platelets Count (16-556):", 16, 556, 286),
        "HMT40": st.sidebar.slider("Hemoglobin level (11-181):", 11, 181, 96),
        "HMT100": st.sidebar.slider("MCH (26-364):", 26, 364, 195),
        "HMT102": st.sidebar.slider("MCHC (32-359):", 32, 359, 195),
        "RCT6": st.sidebar.slider("Urea Nitrogen (1682-115334):", 1682, 115334, 58008),
        "RCT11": st.sidebar.slider("Serum Glucose (7927-234208):", 7927, 234208, 121067),
        "RCT20": st.sidebar.slider("Cholesterol (14306-367317):", 14306, 367317, 190811),
        "RCT392": st.sidebar.slider("Creatinine (6-1923):", 6, 1923, 964),
        "MMSCORE": st.sidebar.slider("Mini-Mental State Exam Score (6-30):", 6, 30, 18),
        "LIMMTOTAL": st.sidebar.slider("Logical Immediate Memory Total (0-23):", 0, 23, 11),
        "LDELTOTAL": st.sidebar.slider("Logical Delayed Memory Total (0-95):", 0, 95, 47)
    })

    return input_dict

def predict(model, scaler, input_df):
    # Scale the input features as during training
    input_features = scaler.transform(input_df)
    prediction = model.predict(input_features)
    probabilities = model.predict_proba(input_features).flatten()
    return prediction, probabilities

def main():
    st.title("Alzheimer's Disease Diagnostic Assistant")
    model, scaler = load_model_and_scaler()
    input_dict = add_sidebar()
    input_df = prepare_input(input_dict)  # Ensure input is properly formatted and typed
    if st.button('Predict'):
        prediction, probabilities = predict(model, scaler, input_df)
        st.write(f"The prediction is: {prediction}")
        st.write("Prediction probabilities:", probabilities)

if __name__ == '__main__':
    main()
