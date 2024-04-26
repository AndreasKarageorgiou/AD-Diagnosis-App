import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_clean_data():
    df = pd.read_csv("AIBL.csv")
  
    # Define diagnosis features and create a target variable
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

df = get_clean_data()

# Visualizing data with histograms
plt.figure(figsize=(12, 6))

# Histogram for examination age
plt.subplot(1, 2, 1)
sns.histplot(data=df, x="ExamAge", kde=True, bins=30)
plt.title("Histogram of Examination Age")

# Assuming MMSCORE exists and is correctly named
plt.subplot(1, 2, 2)
sns.histplot(data=df, x="MMSCORE", kde=True, bins=30)
plt.title("Histogram of MMSCORE")

plt.show()

# Graphical representation of data
plt.figure(figsize=(10, 6))

# Scatter plot for the relationship between age and MMSE
# Assuming MMSCORE is the equivalent to MMSE and 'AgeAtExamination' should be 'ExamAge'
sns.scatterplot(data=df, x="ExamAge", y="MMSCORE", alpha=0.7)
plt.title("Scatter Plot: Relationship Between Age and MMSE")
plt.xlabel("Age at Examination")
plt.ylabel("MMSE")

plt.show()
