import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Φόρτωση dataset
df = pd.read_csv("AIBL.csv")

# Δημιουργία στήλης "Label"
df["Label"] = df["DXAD"].apply(lambda x: 1 if x == "Alzheimer" else 0)

# Υπολογισμός ηλικίας
df["AgeAtExamination"] = df["Examyear"] - df["PTDOBYear"]

# Οπτικοποίηση δεδομένων με ιστογράμματα
plt.figure(figsize=(12, 6))

# Ιστόγραμμα για την ηλικία εξέτασης
plt.subplot(1, 2, 1)
sns.histplot(data=df, x="AgeAtExamination", hue="Label", kde=True, bins=30)
plt.title("Ιστόγραμμα Ηλικίας Εξέτασης")

# Ιστόγραμμα για το MMSE
plt.subplot(1, 2, 2)
sns.histplot(data=df, x="MMSCORE", hue="Label", kde=True, bins=30)
plt.title("Ιστόγραμμα MMSE")

plt.show()

# Γραφηματική αναπαράσταση δεδομένων
plt.figure(figsize=(10, 6))

# Scatter plot για την σχέση ηλικίας και MMSE
sns.scatterplot(data=df, x="AgeAtExamination", y="MMSCORE", hue="Label", alpha=0.7)
plt.title("Scatter Plot: Σχέση Ηλικίας και MMSE")
plt.xlabel("Ηλικία εξέτασης")
plt.ylabel("MMSE")

plt.show()
