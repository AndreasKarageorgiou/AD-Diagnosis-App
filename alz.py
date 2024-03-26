import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ***** PART 1: Data Loading and Exploration *****
# Load dataset
df = pd.read_csv("AIBL.csv")

# Print basic statistics and information about the dataset
print(df.describe()) 
print(df.info())
print(df.isna().sum())

# Create "Label" column indicating Alzheimer's diagnosis
df["Label"] = df["DXAD"].apply(lambda x: 1 if x == "Alzheimer" else 0)

# Calculate age at examination
df["AgeAtExamination"] = df["Examyear"] - df["PTDOBYear"]

# Define the list of desired features
desired_features = ['CDGLOBAL', 'MMSCORE', 'DXMCI', 'LDELTOTAL', 'LIMMTOTAL', 
'AgeAtExamination', 'PTGENDER']

# Extract features and labels
X = df[desired_features]
y = df['DXAD']

# Separate data for Alzheimer and non-Alzheimer patients
alzheimer_df = df[df["Label"] == 1]
non_alzheimer_df = df[df["Label"] == 0]

# Compare age at examination between Alzheimer and non-Alzheimer groups
age_t_test = stats.ttest_ind(alzheimer_df["AgeAtExamination"], non_alzheimer_df["AgeAtExamination"])

# Compare MMSE scores between Alzheimer and non-Alzheimer groups
mmse_t_test = stats.ttest_ind(alzheimer_df["MMSCORE"], non_alzheimer_df["MMSCORE"])

# Calculate Pearson correlation coefficient between age at examination and MMSE
correlation_pearson = stats.pearsonr(df["AgeAtExamination"], df["MMSCORE"])

# Calculate probabilities of belonging to Alzheimer and non-Alzheimer groups
probabilities = df["Label"].value_counts(normalize=True)

# Create dataframe with revised statistics
df_statistics = pd.DataFrame({
    "Feature": ["Age at Examination", "MMSE"],
    "Mean Alzheimer": [alzheimer_df["AgeAtExamination"].mean(), alzheimer_df["MMSCORE"].mean()],
    "Mean Non-Alzheimer": [non_alzheimer_df["AgeAtExamination"].mean(), non_alzheimer_df["MMSCORE"].mean()],
    "t-test p-value (Age)": [age_t_test.pvalue, np.nan],  # NaN for MMSE since it's not applicable
    "t-test p-value (MMSE)": [np.nan, mmse_t_test.pvalue],  # NaN for Age since it's not applicable
    "Pearson Correlation Coefficient": correlation_pearson[0],
    "Probability Alzheimer": probabilities.get(1, 0),
    "Probability Non-Alzheimer": probabilities.get(0, 0),
})

# Save statistics to a CSV file
df_statistics.to_csv("AIBL_statistics.csv")

# Visualize the relationship between age at examination and MMSE
plt.scatter(df["AgeAtExamination"], df["MMSCORE"], c=df["Label"])
plt.xlabel("Age at Examination")
plt.ylabel("MMSE")
plt.show()

# Display a message indicating that statistics have been saved
print("Statistics saved to AIBL_statistics.csv")

# ***** PART 2: Preprocessing *****
# Define a function to preprocess data
def preprocess_data(df, target_column):
    df.dropna(inplace=True)  
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X, y

# Train-Test Split (not included in the revised statistics section)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (not included in the revised statistics section)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ***** PART 3: Model Training and Evaluation *****
# Create models (not included in the revised statistics section)
model_regularized = LogisticRegression(max_iter=10000, C=0.1)
svm_model = SVC()
random_forest_model = RandomForestClassifier()
gradient_boosting_model = GradientBoostingClassifier()
decision_tree_model = DecisionTreeClassifier()
naive_bayes_model = GaussianNB()

# Training and Evaluation Loop (not included in the revised statistics section)
models = [
    ("Regularized Logistic Regression", LogisticRegression(max_iter=10000, C=0.1)),
    ("SVM", SVC()),
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("KNN", KNeighborsClassifier(n_neighbors=10)),
]

# Define 10-fold stratified cross-validation (not included in the revised statistics section)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Loop through models and perform cross-validation (not included in the revised statistics section)
for name, model in models:
    start_time = time.time()

    # Calculate cross-validation scores
    cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy') 

    end_time = time.time()

    # Print results for each model
    print(f"\nModel: {name}")
    print(f"Training Time: {end_time - start_time:.4f} seconds (Overall)")
    print("Cross-Validation Accuracy Scores:", cv_results)
    print(f"Mean Accuracy: {cv_results.mean():.4f}")
    print(f"Standard Deviation: {cv_results.std():.4f}")
