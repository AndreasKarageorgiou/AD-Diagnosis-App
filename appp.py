import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import base64
import plotly.express as px

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
background-image: url("data:image/png;base64,{img}");
background-position: cover; 
background-repeat: repeat;
background-attachment: fixed;
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

# Initialize a session state to store user inputs
if 'input_data' not in st.session_state:
    st.session_state.input_data = pd.DataFrame()

def prepare_user_input(input_dict):
    columns_order = [
        'PTGENDER', 'ExamAge', 'MH16SMOK', 'MH2NEURL', 'MH8MUSCL', 'MHPSYCH', 'MH10GAST',
        'MH4CARD', 'MH9ENDO', 'MH17MALI', 'MH6HEPAT', 'MH12RENA', 'APGEN1', 'APGEN2', 
        'CDGLOBAL', 'MMSCORE', 'LIMMTOTAL', 'LDELTOTAL', 'AXT117', 'BAT126', 'HMT3', 
        'HMT7', 'HMT13', 'HMT40', 'HMT100', 'HMT102', 'RCT6', 'RCT11', 'RCT20', 'RCT392'
    ]
    input_df = pd.DataFrame([input_dict], columns=columns_order)
    return input_df.astype(float)

def add_input_to_data(input_df):
    st.session_state.input_data = pd.concat([st.session_state.input_data, input_df], ignore_index=True)

def main():
    st.title("Alzheimer's Disease Diagnosis Application")

    st.sidebar.header("Input Patient Information")
    st.sidebar.text_input("Patient's Name")
    st.sidebar.text_input("Patient's Nationality")

    # Collect input data
    input_dict = {
        'PTGENDER': st.sidebar.selectbox("Select patient's gender:", [1, 2]),
        'ExamAge': st.sidebar.number_input("Enter patient's age:", min_value=55, max_value=96),
        'MH16SMOK': st.sidebar.radio("Is / Was the Patient Smoking?", [0, 1]),
        'MH2NEURL': st.sidebar.radio("Neurological Assessment (Normal/Abnormal):", [0, 1]),
        'MH8MUSCL': st.sidebar.radio("Musculoskeletal Assessment Normal:", [0, 1]),
        'MHPSYCH': st.sidebar.radio("Does the individual have a positive Mental Health Psych Score?", [0, 1]),
        'MH10GAST': st.sidebar.radio("Is the Gastrointestinal Health within normal limits?", [0, 1]),
        'MH4CARD': st.sidebar.radio("Does the individual have an elevated Cardiovascular Risk?", [0, 1]),
        'MH9ENDO': st.sidebar.radio("Is the Endocrine Function within the normal range?", [0, 1]),
        'MH17MALI': st.sidebar.radio("Is the individual at risk for malignancy?", [0, 1]),
        'MH6HEPAT': st.sidebar.radio("Are there any abnormalities in the Liver Function Marker?", [0, 1]),
        'MH12RENA': st.sidebar.radio("Is the Renal Function within normal limits?", [0, 1]),
        'APGEN1': st.sidebar.slider("APGEN1 value (2-4):", min_value=2, max_value=4),
        'APGEN2': st.sidebar.slider("APGEN2 value (2-4):", min_value=2, max_value=4),
        'CDGLOBAL': st.sidebar.slider("Cognitive Decline Score (0-3):", min_value=0.0, max_value=3.0, step=0.5 ),
        'MMSCORE': st.sidebar.slider("Mini-Mental State Exam Score (6-30):", min_value=6.0, max_value=30.0, step=0.5),
        'LIMMTOTAL': st.sidebar.slider("Logical Immediate Memory Total (0-23):", min_value=0.0, max_value=23.0, step=0.5),
        'LDELTOTAL': st.sidebar.slider("Logical Delayed Memory Total (0-23):", min_value=0.0, max_value=23.0, step=0.5),
        'AXT117': st.sidebar.slider("Thyroid Stimulating Hormone-Blood Test Result (0-13):", min_value=0.0, max_value=13.0, step=0.1),
        'BAT126': st.sidebar.slider("Vitamin B12 level (116-2033):", min_value=116, max_value=2034),
        'HMT3': st.sidebar.slider("RBC Count (3-6):", min_value=3.0, max_value=6.0, step=0.01),
        'HMT7': st.sidebar.slider("WBC Count (2-15):", min_value=2.0, max_value=15.0, step=0.01),
        'HMT13': st.sidebar.slider("Platelets Count (16-556):", min_value=16, max_value=556),
        'HMT40': st.sidebar.slider("Hemoglobin level (9-18):", min_value=9, max_value=18),
        'HMT100': st.sidebar.slider("MCH (21-39):", min_value=21, max_value=39),
        'HMT102': st.sidebar.slider("MCHC (31-36):", min_value=31.0, max_value=36.0, step=0.1),
        'RCT6': st.sidebar.slider("Urea Nitrogen (16-115):", min_value=16, max_value=116),
        'RCT11': st.sidebar.slider("Serum Glucose (46-235):", min_value=46, max_value=235),
        'RCT20': st.sidebar.slider("Cholesterol level (92-367):", min_value=92, max_value=368),
        'RCT392': st.sidebar.slider("Creatinine (0-2):", min_value=0.0, max_value=2.0, step=0.1),
    }
    input_df = prepare_user_input(input_dict)

    if st.sidebar.button("Add Data"):
        add_input_to_data(input_df)
        st.success("Data added!")

    if st.sidebar.button("Visualize Data"):
        # Visualize with a parallel coordinates plot
        fig = px.parallel_coordinates(st.session_state.input_data, color="PTGENDER", labels={col: col for col in st.session_state.input_data.columns})
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
