import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
import numpy as np
import base64
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AD Diagnosis", page_icon="Images/logo1.jpg")  
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("Images/iStock-Credit-Nobi_Prizue-1200-628-5-17-23.jpg")
img1 = get_img_as_base64("Images/ImageForNews_760599_1696326718990377.webp")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img1}");
background-size: cover;
background-position: center; 
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img1}");
background-position: top-right; 
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


def load_resources():
    model = joblib.load("Model/model.joblib")
    scaler = joblib.load("Model/scaler.joblib")
    encoder = joblib.load('Model/label_encoder.joblib')
    return model, scaler, encoder

# Function to prepare user input for prediction
def get_clean_data():
    df = pd.read_csv("AIBL.csv")
    # Apply any necessary transformations as during training
    diagnosis_features = ['DXNORM', 'DXMCI', 'DXAD']
    def create_DXTYPE(DXNORM, DXMCI, DXAD):
        if DXNORM == 1:
            return 0
        elif DXMCI == 1:
            return 1
        elif DXAD == 1:
            return 2
        else:
            return -1

    # Apply the function to create a target variable 'DXTYPE'
    df['DXTYPE'] = df.apply(lambda row: create_DXTYPE(row['DXNORM'], row['DXMCI'], row['DXAD']), axis=1)

    # Calculate age before dropping 'Examyear' and 'PTDOBYear'
    if 'Examyear' in df.columns and 'PTDOBYear' in df.columns:
        df["ExamAge"] = df["Examyear"] - df["PTDOBYear"]    

    # Columns to drop including the 'APTyear', 'Examyear', 'PTDOBYear', and 'DXCURREN'
    columns_to_drop = diagnosis_features + ['APTyear', 'Examyear', 'PTDOBYear', 'DXCURREN']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    return df

def prepare_user_input(input_dict):
    # Ensure the features are in the exact order and naming as during model training
    columns_order = [
        'APGEN1', 'APGEN2', 'CDGLOBAL', 'AXT117', 'BAT126', 'HMT3', 'HMT7',
        'HMT13', 'HMT40', 'HMT100', 'HMT102', 'RCT6', 'RCT11', 'RCT20', 'RCT392',
        'MHPSYCH', 'MH2NEURL', 'MH4CARD', 'MH6HEPAT', 'MH8MUSCL', 'MH9ENDO',
        'MH10GAST', 'MH12RENA', 'MH16SMOK', 'MH17MALI', 'MMSCORE', 'LIMMTOTAL',
        'LDELTOTAL', 'PTGENDER', 'ExamAge']
    # Create a DataFrame ensuring the columns are correctly ordered
    input_df = pd.DataFrame([input_dict], columns=columns_order)
    return input_df.astype(float)

# Function to perform prediction and display results
def perform_prediction(input_df, model, scaler, encoder):
    # Scale the input features as during training
    scaled_features = scaler.transform(input_df)
    # Predict
    predictions = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[0]
    # Decode prediction
    prediction_label = encoder.inverse_transform([predictions])[0]
    # Display results
    st.subheader("Diagnosis Result")
    st.write(f"The prediction is: {prediction_label}")
    st.write("Prediction probabilities:")
    for i, label in enumerate(encoder.classes_):
        st.write(f"{label}: {probabilities[i]:.2f}")


def display_home_page():
    st.subheader("Welcome to the Home Page")
    st.write("This is the landing page of your application.")
    st.write("Click on 'Questionnaire' in the sidebar to start the questionnaire.")
    st.subheader("A WebApp Created By Andreas Karageorgiou P2018122")

    # Display contact information and current local time
    st.markdown("---")
    st.subheader("Contact Information")
    st.write("Name: Andreas Karageorgiou")
    st.write("Location: Your Location")
    st.write("Phone: Your Phone Number")
    

    # Add a button to initiate the questionnaire
    if st.button("Let's get to the questionnaire"):
        display_questionnaire_page()
        return  # Exit the function to prevent further execution

def display_about_page():
    st.title("About Us")

def main():
    st.title("Alzheimer's Disease Diagnosis Application, Powered by the Ionian University")
    

    st.sidebar.subheader("Alzheimer's disease presents a pressing challenge")
    st.sidebar.header("App Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Questionnaire", "About Us"])

    if page == "Home":
        display_home_page()
    elif page == "Questionnaire":
        display_questionnaire_page()
    elif page == "About Us":
        display_about_page()

    st.sidebar.subheader("A WebApp Created By Andreas Karageorgiou")

    

def display_questionnaire_page():
    st.subheader("Questionnaire")
      
    st.header("Demographics")
    st.write("__________________________")
    st.write("")

    # Question 1: Patients Full Name
    st.write("1. Patients Full Name")
    patient_name = st.text_input("Enter patient's full name:", key="patient_name")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 2: Patients Nationality
    st.write("2. Patients Nationality")
    patient_nationality = st.text_input("Enter patient's nationality:", key="patient_nationality")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 3: Patients Gender
    st.write("3. Patients Gender")
    patient_gender = st.selectbox("Select patient's gender:", options=["Male", "Female", "Other"], key="patient_gender")
    
    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")
    
    # Question 4: Patients Age
    st.write("4. Patients Age")
    patient_age = st.number_input("Enter patient's age:", min_value=50, max_value=110, key="patient_age")

    
    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    st.subheader("Medical History")
    st.write("__________________________")
    st.write("")

    # Question 5: Is / Was the Patient Smoking?
    st.write("5. Is / Was the Patient Smoking?MH16SMOK")
    patient_smoking = st.radio("Select option:", options=["Yes", "No"], key="smoking")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 6: Neurological Assessment
    st.write("6. Neurological Assessment (MH2NEURL)")
    neurological_assessment = st.radio("Select option:", options=["Normal", "Abnormal"], key="neurological_assessment")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 7: Is the Musculoskeletal Assessment within the normal range?
    st.write("7. Is the Musculoskeletal Assessment within the normal range?MH8MUSCL ")
    musculoskeletal_assessment = st.radio("Select option:", options=["No", "Yes"], key="musculoskeletal")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 8: Does the individual have a positive Mental Health Psychometric Score?
    st.write("8. Does the individual have a positive Mental Health Psychometric Score?MHPSYCH ")
    mental_health_score = st.radio("Select option:", options=["No", "Yes"], key="mental_health")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 9: Is the Gastrointestinal Health within normal limits?
    st.write("9. Is the Gastrointestinal Health within normal limits? MH10GAST")
    gastrointestinal_health = st.radio("Select option:", options=["No", "Yes"], key="gastrointestinal")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 10: Does the individual have an elevated Cardiovascular Risk?
    st.write("10. Does the individual have an elevated Cardiovascular Risk? MH4CARD")
    cardiovascular_risk = st.radio("Select option:", options=["No", "Yes"], key="cardiovascular")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 11: Is the Endocrine Function within the normal range?
    st.write("11. Is the Endocrine Function within the normal range? MH9ENDO")
    endocrine_function = st.radio("Select option:", options=["No", "Yes"], key="endocrine")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 12: Is the individual at risk for malignancy?
    st.write("12. Is the individual at risk for malignancy?MH17MALI ")
    malignancy_risk = st.radio("Select option:", options=["No", "Yes"], key="malignancy")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 13: Are there any abnormalities in the Liver Function Marker?
    st.write("13. Are there any abnormalities in the Liver Function Marker?MH6HEPAT ")
    liver_function_abnormalities = st.radio("Select option:", options=["No", "Yes"], key="liver")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 14: Is the Renal Function within normal limits?
    st.write("14. Is the Renal Function within normal limits? MH12RENA")
    renal_function = st.radio("Select option:", options=["No", "Yes"], key="renal")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")
    
    st.subheader("Apolipoprotein E (ApoE)")
    st.write("__________________________")
    st.write("")

    # Question 15: APGEN1
    st.write("15. APGEN1")
    apgen1_option = st.radio("Select option:", options=[2, 3, 4], key="apgen1")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 16: APGEN2
    st.write("16. APGEN2")
    apgen2_option = st.radio("Select option:", options=[2, 3, 4], key="apgen2")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    st.subheader("Cognitive Tests")
    st.write("__________________________")
    st.write("")

    # Question 17: CDGLOBAL - Cognitive Decline Score
    st.write("17. CDGLOBAL - Cognitive Decline Score")
    cdglobal_option = st.slider("Select a value:", min_value=0, max_value=5, key="cdglobal")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 18: LDELTOTAL - Logical Delayed Memory Total
    st.write("18. LDELTOTAL - Logical Delayed Memory Total")
    ldeltotal_option = st.slider("Select a value:", min_value=0, max_value=95, key="ldeltotal")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 19: MMSCORE - Mini-Mental State Exam Score
    st.write("19. MMSCORE - Mini-Mental State Exam Score")
    mmscore_option = st.slider("Select a value:", min_value=6, max_value=30, key="mmscore")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 20: LIMMTOTAL - Logical Immediate Memory Total
    st.write("20. LIMMTOTAL - Logical Immediate Memory Total")
    limmtotal_option = st.slider("Select a value:", min_value=0, max_value=23, key="limmtotal")

    st.subheader("Blood Tests")
    st.write("__________________________")
    st.write("")

    # Question 21: AXT117 - Thyroid Stimulating Hormone-Blood Test Result (Marker 117)
    st.write("21. AXT117 - Thyroid Stimulating Hormone-Blood Test Result (Marker 117)")
    axt117_option = st.slider("Select a value:", min_value=0, max_value=1300, key="axt117")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 22: BAT126 - Vitamin B12
    st.write("22. BAT126 - Vitamin B12")
    bat126_option = st.slider("Select a value:", min_value=100, max_value=2500, key="bat126")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 23: HMT100 - MCH (Mean Corpuscular Hemoglobin)
    st.write("23. HMT100 - MCH (Mean Corpuscular Hemoglobin)")
    hmt100_option = st.slider("Select a value:", min_value=26, max_value=364, key="hmt100")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 24: HMT102 - MCHC (Mean Corpuscular Hemoglobin Concentration)
    st.write("24. HMT102 - MCHC (Mean Corpuscular Hemoglobin Concentration)")
    hmt102_option = st.slider("Select a value:", min_value=32, max_value=359, key="hmt102")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 25: HMT13 - Platelets
    st.write("25. HMT13 - Platelets")
    hmt13_option = st.slider("Select a value:", min_value=16, max_value=556, key="hmt13")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 26: HMT3 - RBC (Red Blood Cell Count)
    st.write("26. HMT3 - RBC (Red Blood Cell Count)")
    hmt3_option = st.slider("Select a value:", min_value=5, max_value=584, key="hmt3")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 27: HMT40 - Hemoglobin
    st.write("27. HMT40 - Hemoglobin")
    hmt40_option = st.slider("Select a value:", min_value=11, max_value=181, key="hmt40")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 28: HMT7 - WBC (White Blood Cell Count)
    st.write("28. HMT7 - WBC (White Blood Cell Count)")
    hmt7_option = st.slider("Select a value:", min_value=3, max_value=145, key="hmt7")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 29: RCT11 - Serum Glucose
    st.write("29. RCT11 - Serum Glucose")
    rct11_option = st.slider("Select a value:", min_value=7927, max_value=234208, key="rct11")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 30: RCT20 - Cholesterol (High Performance)
    st.write("30. RCT20 - Cholesterol (High Performance)")
    rct20_option = st.slider("Select a value:", min_value=14306, max_value=367317, key="rct20")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 31: RCT392 - Creatinine (Rate Blanked)
    st.write("31. RCT392 - Creatinine (Rate Blanked)")
    rct392_option = st.slider("Select a value:", min_value=6, max_value=1923, key="rct392")

    # Double line separator
    st.write("")
    st.write("__________________________")
    st.write("")

    # Question 32: RCT6 - Urea Nitrogen
    st.write("32. RCT6 - Urea Nitrogen")
    rct6_option = st.slider("Select a value:", min_value=1682, max_value=115334, key="rct6")
    

    submit_button = st.form_submit_button(label='Predict')
        

   


if __name__ == "__main__":
    main()