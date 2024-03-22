from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import pandas as pd


# Load the CSV data
df = pd.read_csv('AIBL.csv')

# Define your complete feature list
desired_features = ['APGEN1','APGEN2','CDGLOBAL','AXT117','BAT126','HMT3','HMT7',
'HMT13','HMT40','HMT100','HMT102','RCT6','RCT11','RCT20','RCT392',
'MHPSYCH','MH2NEURL','MH4CARD','MH6HEPAT','MH8MUSCL','MH9ENDO',
'MH10GAST','MH12RENA','MH16SMOK','MH17MALI','MMSCORE','LIMMTOTAL',
'LDELTOTAL','DXCURREN','DXNORM','DXMCI' ,'PTGENDER','Examyear',
'APTyear','PTDOBYear']

X = df[desired_features]
y = df['DXAD']  # Assuming 'Label' is your target column

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
