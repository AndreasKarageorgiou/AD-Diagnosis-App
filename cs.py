import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
import numpy as np
import base64
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import time

st.set_page_config(page_title="AD Diagnosis", page_icon="Images/trans-logo.png", layout="wide")  
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("Images/bluehead1.png")
img1 = get_img_as_base64("Images/bluehead1.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img1}");
background-size: cover;
background-position: right; 
background-repeat: no-repeat;
background-attachment: fixed;
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


def display_home_page():
    st.subheader("Welcome to Our Alzheimer's Disease Diagnostic Assistant")

    # Use markdown to add a visual separator and subtitle
    st.markdown("---")
    st.header("Discover the Early Signs of Alzheimer's")

    # Column layout for text and image

    st.write("""
        Our diagnostic tool helps in the early detection of Alzheimer's disease by analyzing responses to a series of medical and health-related questions. Early detection can significantly impact the management of the condition, allowing for early intervention strategies and better planning.
        """)
    
    st.markdown("---")
    st.subheader("The Only thing you need to do is to...")
    st.write("""
        Simply follow the prompts to answer questions about health status, medical history, and lifestyle. Our advanced algorithms will analyze your inputs and provide an assessment based on current medical knowledge.
        """)

    st.markdown("---")


    
    # Navigation buttons within a single row
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("### Are you ready?")

        if st.button("Let's get to the Test"):
            st.session_state['page'] = "Questionnaire"

    with col3:
        st.markdown("### Need More Information?")

        if st.button("Learn More About Us"):
            st.session_state['page'] = "About Us"
    
    with col1:
        st.markdown("### How it Works ?")

        with st.expander("Learn More About Alzheimer's Detection"):
            st.write("""
Alzheimer's disease is a progressive neurodegenerative disorder that affects memory, thinking, and behavior. Early detection of Alzheimer's can be crucial for several reasons:

Treatment Opportunities: While there's no cure for Alzheimer's, early detection can provide access to treatment options that may help to alleviate symptoms or slow the disease's progression.

Planning for the Future: Individuals diagnosed early may have more time to plan for their future while they still have the capacity to make important decisions regarding care, living arrangements, and financial and legal matters.

Lifestyle Adjustments: Patients can adopt lifestyle changes that may help manage the disease better, such as diet, exercise, cognitive training, and social engagement.

Support Systems: It allows patients and their families to establish support systems, including joining support groups and accessing services and resources.
                     
Clinical Trials: Patients may have the opportunity to participate in clinical trials, contributing to the research and development of new treatments.

Our application aims to contribute to this goal by providing a tool that can assist in the preliminary screening process. By answering a series of targeted questions, users can assess their or their loved one’s risk for Alzheimer's. The app uses data analytics and machine learning models trained on medical data to predict the likelihood of Alzheimer's disease. While not a replacement for professional medical diagnosis and advice, our app offers a convenient and accessible way to stay informed about one's cognitive health.

The app is designed with user-friendliness in mind, ensuring that the questions are easy to understand and the results are presented clearly. We believe that technology can be a powerful ally in the fight against Alzheimer's, offering new ways to approach detection and management of the disease.        """)
  
    
def display_about_page():
    st.title("About Us")
    st.markdown("---")

    # Team Section
    col1, col2, = st.columns(2)
    with col1:
        st.header("Meet The Creator")
        st.image("Images/CV photo.jpg", caption="Founder")
        st.subheader("ANDREAS KARAGEORGIOU")
        st.write("Student at the Ionian University .")

    with col2:
        st.header("Our Mission")
        st.write("The application is driven by a commitment to enhancing early diagnosis and intervention strategies for Alzheimer's disease, a condition that significantly impacts millions of individuals and their families worldwide. The goal is to provide an accessible, user-friendly tool that leverages advanced data analytics to predict an individual's likelihood of having Alzheimer's based on various health indicators and test results. This application aims to bridge the gap between the latest research in cognitive health and practical, everyday use by medical professionals and potentially at-risk populations.")
        st.write("The creation of this project was inspired by the personal experiences of the developer, who witnessed a beloved family member struggle with Alzheimer's disease. The journey of seeing their relative's gradual decline, coupled with the challenges in obtaining an early and clear diagnosis, highlighted the need for more accessible diagnostic tools. This personal connection to the issue fueled a passion for making a difference in the lives of those affected by Alzheimer's.")
        st.write("By integrating a scientifically-backed approach with user-friendly technology, the application aspires to empower users with knowledge and actionable insights, thereby fostering earlier interventions that can significantly alter the disease's impact. The hope is that by making these tools widely available, more families can have precious time and better outcomes, potentially slowing the progression of symptoms through timely and tailored interventions.")
    
    st.markdown("---")

    
    # Contact Information
    st.header("Contact Us")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("p18kara@ionio.gr")
    with col2:
        st.write("Location: Athira 58005, Pella")
    with col3:
        st.write("Phone: +30 698 069 0607")
    

    # Footer or Closing Remark
    st.markdown("---")

    st.subheader("Thank You for Visiting!")
    st.write("We appreciate your interest and hope to continue providing value through our work.")


def perform_prediction(input_df, model, scaler, encoder, name, nationality):
    # Scale the input features as during training
    scaled_features = scaler.transform(input_df)
    # Predict
    predictions = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[0]
    # Decode prediction
    prediction_label = encoder.inverse_transform([predictions])[0]
    # Display results
    diagnosis_messages = {
        0: "Normal",
        1: "MCI",
        2: "Alzheimer's Disease"
    }
    
    # Get custom message based on prediction
    custom_message = diagnosis_messages.get(predictions[0], "Unknown Diagnosis")
    
    # Display results
    result_message = f"""
    BASED ON THE INFORMATION SUBMITTED FOR:
    
            {name.title()},  {nationality.title()},

            THE PREDICTION IS: {custom_message}
    """
    st.subheader("Diagnosis Result")
    st.write(result_message)
    st.write("Prediction probabilities:")
    for i, label in enumerate(encoder.classes_):
        st.write(f"{label}: {probabilities[i]:.2f}")


def main():

    st.subheader("Alzheimer's Disease Diagnosis App, Powered by the Ionian University")
    st.markdown("---")
    st.sidebar.markdown("---")
    st.sidebar.header("App Navigation")
    
    if 'page' not in st.session_state:
        st.session_state['page'] = "Home"

    # Set up sidebar for navigation
    page = st.sidebar.radio("Go to", ["Home", "Questionnaire", "About Us"], index=["Home", "Questionnaire", "About Us"].index(st.session_state['page']))
    st.sidebar.markdown("---")


    # Update the session state based on sidebar interaction
    if page != st.session_state['page']:
        st.session_state['page'] = page

    # Display pages based on current state
    if st.session_state['page'] == "Home":
        display_home_page()
    elif st.session_state['page'] == "Questionnaire":
        display_questionnaire_page()
    elif st.session_state['page'] == "About Us":
        display_about_page()

    st.sidebar.markdown("---")
    st.sidebar.subheader("A WebApp Created By Andreas Karageorgiou")


def display_questionnaire_page():
    custom_css = """
        <style>
            div.stSelectbox > div {
                width: 250px;  /* Adjust the width as needed */
            }
    
            div.stTextInput > div {
                width: 250px;  /* Adjust the width as needed */
        }
            div.stNumberInput > div {
                    width: 250px;  /* Adjust the width as needed */
                
        }
            div.stSlider > div {
                    width: 600px;  /* Adjust the width as needed */
        }
        </style>
        """
    st.markdown(custom_css, unsafe_allow_html=True)
    model, scaler, encoder = load_resources()
    yes_no_mapping = {"No": 0, "Yes": 1}
    gender_mapping = {"Male": 1, "Female": 2}

    name = st.text_input("1.Enter patient's full name:")
    nationality = st.text_input("2.Enter patient's nationality:")

    
    input_dict = {
        'PTGENDER': gender_mapping[st.selectbox("3.Select patient's gender:", ["Male", "Female"])],
        'ExamAge': st.number_input("4.Enter patient's age:", min_value=55, max_value=96),
        'MH16SMOK': yes_no_mapping[st.radio("5.Is / Was the Patient Smoking?", ["No", "Yes"], horizontal=True)],
        'MH2NEURL': yes_no_mapping[st.radio("6.Neurological Assessment (Normal/Abnormal):", ["No", "Yes"], horizontal=True)],
        'MH8MUSCL': yes_no_mapping[st.radio("7.Musculoskeletal Assessment Normal:", ["No", "Yes"], horizontal=True)],
        'MHPSYCH': yes_no_mapping[st.radio("8.Does the individual have a positive Mental Health Psych Score?", ["No", "Yes"], horizontal=True)],
        'MH10GAST': yes_no_mapping[st.radio("9.Is the Gastrointestinal Health within normal limits?", ["No", "Yes"], horizontal=True)],
        'MH4CARD': yes_no_mapping[st.radio("10.Does the individual have an elevated Cardiovascular Risk?", ["No", "Yes"], horizontal=True)],
        'MH9ENDO': yes_no_mapping[st.radio("11.Is the Endocrine Function within the normal range?", ["No", "Yes"], horizontal=True)],
        'MH17MALI': yes_no_mapping[st.radio("12.Is the individual at risk for malignancy?", ["No", "Yes"], horizontal=True)],
        'MH6HEPAT': yes_no_mapping[st.radio("13.Are there any abnormalities in the Liver Function Marker?", ["No", "Yes"], horizontal=True)],
        'MH12RENA': yes_no_mapping[st.radio("14.Is the Renal Function within normal limits?", ["No", "Yes"], horizontal=True)],
        'APGEN1': st.radio("15.APGEN1 value :",[ "2", "3", "4"], horizontal=True),
        'APGEN2': st.radio("15.APGEN2 value ):",[ "2", "3", "4"], horizontal=True),
        'CDGLOBAL': st.slider("17.Cognitive Decline Score (0-3):", min_value=0.0, max_value=3.0, step=0.5 ),
        'MMSCORE': st.slider("18.Mini-Mental State Exam Score (6-30):", min_value=6.0, max_value=30.0, step=0.5),
        'LIMMTOTAL': st.slider("19.Logical Immediate Memory Total (0-23):", min_value=0.0, max_value=23.0, step=0.5),
        'LDELTOTAL': st.slider("20.Logical Delayed Memory Total (0-23):", min_value=0.0, max_value=23.0, step=0.5),
        'AXT117': st.slider("21.Thyroid Stimulating Hormone-Blood Test Result (0-13):", min_value=0.0, max_value=13.0, step=0.1),
        'BAT126': st.slider("22.Vitamin B12 level (116-2034):", min_value=116, max_value=2034),
        'HMT3': st.slider("23.RBC Count (3-6):", min_value=3.0, max_value=6.0, step=0.01),
        'HMT7': st.slider("24.WBC Count (2-15):", min_value=2.0, max_value=15.0, step=0.01),
        'HMT13': st.slider("25.Platelets Count (16-556):", min_value=16, max_value=556),
        'HMT40': st.slider("26.Hemoglobin level (9-18):", min_value=9, max_value=18),
        'HMT100': st.slider("27.MCH (21-39):", min_value=21, max_value=39),
        'HMT102': st.slider("28.MCHC (31-36):", min_value=31.0, max_value=36.0, step=0.1),
        'RCT6': st.slider("29.Urea Nitrogen (16-115):", min_value=16, max_value=116),
        'RCT11': st.slider("30.Serum Glucose (46-235):", min_value=46, max_value=235),
        'RCT20': st.slider("31.Cholesterol level (92-368):", min_value=92, max_value=368),
        'RCT392': st.slider("32.Creatinine (0-2):", min_value=0.0, max_value=2.0, step=0.1),
    }
    input_df = prepare_user_input(input_dict)

    if st.button("Predict"):
        progress_bar = st.progress(0)
        for perc_completed in range(100):
            time.sleep(0.05)
            progress_bar.progress(perc_completed+1)

        st.success("Your data have been uploaded")  

        with st.expander("Click Here to See your Results"):
            perform_prediction(input_df, model, scaler, encoder, name, nationality)


st.sidebar.image("Images/transpar.png", caption="Early Detection Makes a Difference")    

if __name__ == "__main__":
    main()