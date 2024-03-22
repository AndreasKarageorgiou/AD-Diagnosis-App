import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("AIBL.csv")

# Create "Label" column
df["Label"] = df["DXAD"].apply(lambda x: 1 if x == "Alzheimer" else 0)

# Calculate age
df["AgeAtExamination"] = df["Examyear"] - df["PTDOBYear"]

# Data Preprocessing
df.dropna(inplace=True)

# Define your updated feature list
desired_features = ['CDGLOBAL', 'MMSCORE', 'DXMCI', 'LDELTOTAL', 'LIMMTOTAL', 
'AgeAtExamination', 'PTGENDER']

# Extract features and labels
X = df[desired_features]
y = df['DXAD']  

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models
model_regularized = LogisticRegression(max_iter=10000, C=0.1)
svm_model = SVC()
random_forest_model = RandomForestClassifier()
gradient_boosting_model = GradientBoostingClassifier()
decision_tree_model = DecisionTreeClassifier()
naive_bayes_model = GaussianNB()

# Training and Evaluation Loop
models = [
    ("Regularized Logistic Regression", LogisticRegression(max_iter=10000, C=0.1)),
    ("SVM", SVC()),
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("KNN", KNeighborsClassifier(n_neighbors=10)),
]

# Define 10-fold stratified cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Loop through models and perform cross-validation
for name, model in models:
    start_time = time.time()

    # Calculate cross-validation scores
    cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy') 

    end_time = time.time()

    # Print results for the model
    print(f"\nModel: {name}")
    print(f"Training Time: {end_time - start_time:.4f} seconds (Overall)")
    print("Cross-Validation Accuracy Scores:", cv_results)
    print(f"Mean Accuracy: {cv_results.mean():.4f}")
    print(f"Standard Deviation: {cv_results.std():.4f}") 

import joblib

# Train your random forest model
random_forest_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(random_forest_model, 'random_forest_model.pkl')