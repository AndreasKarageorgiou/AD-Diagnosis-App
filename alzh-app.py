# Load the trained Random Forest model
import joblib
import pandas as pd


model = joblib.load('random_forest_model.pkl')

# Define questions for user input
questions = {
    'CDGLOBAL': 'Enter CDGLOBAL value (0-100): ',
    'MMSCORE': 'Enter MMSCORE value (0-30): ',
    'DXMCI': 'Enter DXMCI value (0 or 1): ',
    'LDELTOTAL': 'Enter LDELTOTAL value (0-100): ',
    'LIMMTOTAL': 'Enter LIMMTOTAL value (0-100): ',
    'AgeAtExamination': 'Enter Age at Examination: ',
    'PTGENDER': 'Enter PTGENDER (M/F): '
}
# Load dataset
df = pd.read_csv("AIBL.csv")

# Get user input
user_data = {}
for feature, question in questions.items():
    user_input = input(question)
    user_data[feature] = user_input

# Prepare user input for prediction
user_df = pd.DataFrame([user_data])

# Make prediction
prediction = model.predict(user_df)
probability = model.predict_proba(user_df)[0][1]

# Display prediction
if prediction == 1:
    print(f'The predicted probability of Alzheimer\'s disease is {probability:.2f}')
else:
    print(f'The predicted probability of Alzheimer\'s disease is {probability:.2f}')
