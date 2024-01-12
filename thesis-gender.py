import pandas as pd

# Ανοίξτε το σύνολο δεδομένων
df = pd.read_csv("AIBL_metadata.csv")

# Ομαδοποιήστε τα δεδομένα με βάση το φύλο
grouped = df.groupby("Gender")

# Υπολογίστε τη συσχέτιση μεταξύ κάθε χαρακτηριστικού και της διάγνωσης της νόσου του Alzheimer για κάθε ομάδα
correlations = grouped["Label"].corr()

# Δημιουργήστε έναν πίνακα συσχετίσεων που να ομαδοποιεί τις συσχετίσεις ανάλογα με το φύλο
pivot_table = correlations.pivot_table(values="Label", index="Feature", columns="Gender")
