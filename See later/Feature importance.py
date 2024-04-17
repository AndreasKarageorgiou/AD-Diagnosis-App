from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import pandas as pd


# Load dataset
df = pd.read_csv("AIBL.csv")

# Define diagnosis features
diagnosis_features = ['DXNORM', 'DXMCI', 'DXAD', 'DXCURREN']

def create_DXTYPE(DXNORM, DXMCI, DXAD, DXCURREN):
    if DXNORM == 1:
        return 0  # Represent "normal" 
    elif DXMCI == 1:
        return 1  # Represent "MCI"
    elif DXAD == 1:
        return 2  # Represent "AD"
    else:
        return 3  # Handle the case where none of the conditions are satisfied

df['DXTYPE'] = df.apply(lambda row: create_DXTYPE(row['DXNORM'], row['DXMCI'], row['DXAD'], row['DXCURREN']), axis=1)

# Now remove the unnecessary diagnosis features:
df.drop(diagnosis_features, axis=1, inplace=True)

# Calculate age
df["ExamAge"] = df["Examyear"] - df["PTDOBYear"]

# Data Preprocessing
df.dropna(inplace=True)

# Define your updated feature list
desired_features = ['APGEN1','APGEN2','CDGLOBAL','AXT117','BAT126','HMT3','HMT7' ,
'HMT13','HMT40','HMT100','HMT102','RCT6','RCT11','RCT20','RCT392', 'MHPSYCH',
'MH2NEURL','MH4CARD','MH6HEPAT','MH8MUSCL','MH9ENDO',
'MH10GAST','MH12RENA','MH16SMOK','MH17MALI','MMSCORE','LIMMTOTAL', 
'LDELTOTAL' ,'PTGENDER',"ExamAge", 'APTyear']

# Extract features and labels
X = df[desired_features]
y = df['DXTYPE']

# Train a Random Forest (assuming your data is loaded correctly)
model = RandomForestClassifier()
model.fit(X, y)

# Get Raw Importances
raw_importances = model.feature_importances_

# Normalize to percentage
normalized_importances = 100 * (raw_importances / raw_importances.sum())

# Print feature names and their normalized importance percentage
for feature, importance in zip(desired_features, normalized_importances):
    print(f"Feature: {feature}, Importance: {importance:.2f}%") 
