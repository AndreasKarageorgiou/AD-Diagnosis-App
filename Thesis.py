import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
# import seaborn as sns


# Ανοίξτε το σύνολο δεδομένων
df = pd.read_csv("AIBL.csv")

# Υπολογίστε τα στατιστικά

# Σύνολικά χαρακτηριστικά
mean = df.mean()
std = df.std()
max = df.max()
min = df.min()
distribution = df.hist()

# Μέσα για κάθε διάγνωση
mean_alzheimer = df[df["Label"] == "Alzheimer"].mean()
mean_non_alzheimer = df[df["Label"] == "Non-Alzheimer"].mean()

# Συσχέτιση μεταξύ των χαρακτηριστικών
correlation_pearson = stats.pearsonr(df["Age"], df["MMSE"])
correlation_spearman = stats.spearmanr(df["Age"], df["MMSE"])

# Πιθανότητες διάγνωσης
probabilities = df["Label"].value_counts() / len(df)

# Αποθηκεύστε τα στατιστικά σε ένα dataframe
df_statistics = pd.DataFrame({
    "Characteristic": df.columns,
    "Mean": mean,
    "Standard deviation": std,
    "Maximum": max,
    "Minimum": min,
    "Distribution": distribution,
    "Mean Alzheimer": mean_alzheimer,
    "Mean non-Alzheimer": mean_non_alzheimer,
    "Pearson correlation coefficient": correlation_pearson,
    "Spearman correlation coefficient": correlation_spearman,
    "Probability of Alzheimer": probabilities[0],
    "Probability of non-Alzheimer": probabilities[1]
})

# Αποθηκεύστε το dataframe σε ένα αρχείο CSV
df_statistics.to_csv("AIBL_statistics.csv")