import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime



# Load the trained XGBoost model
model_file = "best_xgb_model.pkl"
model = joblib.load(model_file)


# Define the list of desired features
desired_features = ['APGEN1', 'APGEN2', 'CDGLOBAL', 'AXT117', 'BAT126', 'HMT3', 'HMT7',
'HMT13', 'HMT40', 'HMT100', 'HMT102', 'RCT6', 'RCT11', 'RCT20', 
'RCT392', 'MHPSYCH', 'MH2NEURL', 'MH4CARD', 'MH6HEPAT', 'MH8MUSCL',
'MH9ENDO', 'MH10GAST', 'MH12RENA', 'MH16SMOK', 'MH17MALI', 
'MMSCORE', 'LIMMTOTAL', 'LDELTOTAL', 'PTGENDER', 'ExamAge', 'APTyear']


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://d2jx2rerrg6sh3.cloudfront.net/images/news/ImageForNews_760599_1696326718990377.jpg");
background-size: cover;
background-position: top right;
background-repeat: no-repeat;
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
st.sidebar.subheader("Alzheimer's disease presents a pressing challenge")
st.sidebar.subheader("A WebApp Created By Andreas Karageorgiou")

def main():
    st.title("Alzheimer's Disease Diagnosis Application")
    st.sidebar.header("App Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Questionnaire"])

    if page == "Home":
        display_home_page()
    elif page == "Questionnaire":
        display_questionnaire_page()

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

def display_questionnaire_page():
    st.subheader("Questionnaire")
    st.write("This is the questionnaire page.")
    st.write("Your questionnaire code goes here.")

def display_questionnaire_page():
        
    st.header("Demographics")
    st.write("__________________________")
    st.write("")

    # Question 1: Patients Full Name
    st.write("1. Patients Full Name")
    patient_name = st.text_input("Enter patient's full name:", key="patient_name")

    # Question 2: Patients Nationality
    st.write("2. Patients Nationality")
    patient_nationality = st.text_input("Enter patient's nationality:", key="patient_nationality")

    # Question 3: Patients Gender
    st.write("3. Patients Gender")
    patient_gender = st.selectbox("Select patient's gender:", options=["Male", "Female", "Other"], key="patient_gender")

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
    pass

   


if __name__ == "__main__":
    main()