import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("AIBL.csv")

# Calculate age
df["AgeAtExamination"] = df["Examyear"] - df["PTDOBYear"]

# Data Preprocessing
df.dropna(inplace=True)

# Define features of interest
desired_features = ['CDGLOBAL', 'MMSCORE', 'DXMCI', 'LDELTOTAL', 'LIMMTOTAL', 'AgeAtExamination', 'PTGENDER']

# Select subset of features
df_subset = df[desired_features + ['DXAD']]

# Compute the correlation matrix
corr_matrix = df_subset.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features with DXAD")
plt.show()
