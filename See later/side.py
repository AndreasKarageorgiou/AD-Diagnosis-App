import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import base64

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

# Function to load the model, scaler, and label encoder
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



# Main function to handle the Streamlit app layout
def main():
    st.title("Alzheimer's Disease Diagnosis Application")
    
    model, scaler, encoder = load_resources()

    # Adding a sidebar for user input
    st.sidebar.header("Input Patient Information")
    st.sidebar.text_input("Patient's Name")

    st.sidebar.text_input("Patient's Nationality")

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

    if st.sidebar.button("Predict"):
        progress_bar = st.progress(0)
        for perc_completed in range(100):
            time.sleep(0.05)
            progress_bar.progress(perc_completed+1)

        st.success("Your data have been uploaded")  

        with st.expander("Click Here to See your Results"):
            perform_prediction(input_df, model, scaler, encoder)

if __name__ == '__main__':
    main()
