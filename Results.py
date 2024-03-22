import pandas as pd
import numpy as np
import time
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
models = {
    "Regularized Logistic Regression": LogisticRegression(max_iter=10000, C=0.1),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
}

# Define 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results dictionary
results = {}

# Evaluate models
for model_name, model in models.items():
    start_time = time.time()
    
    if model_name == "KNN":
        # KNN requires separate handling due to its training and prediction time
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_start_time = time.time()
        knn_model.fit(X_train_scaled, y_train)
        knn_train_time = time.time() - knn_start_time
        
        knn_pred_start_time = time.time()
        knn_model.predict(X_test_scaled)
        knn_pred_time = time.time() - knn_pred_start_time

        results[model_name + " Model Training Time"] = knn_train_time
        results[model_name + " Model Prediction Time"] = knn_pred_time
    else:
        cv_results = cross_val_score(model, X, y, cv=cv)
        results[model_name + " Cross-Validation Results"] = cv_results
        results[model_name + " Mean Accuracy"] = cv_results.mean()

    elapsed_time = time.time() - start_time
    results[model_name + " Time taken"] = elapsed_time

    # Print results
    print(f"{model_name} Cross-Validation Results:")
    print(cv_results)
    print(f"Mean Accuracy: {cv_results.mean()}")
    print(f"Time taken: {elapsed_time} seconds\n")

# Print and save results to a file
with open("model_results.txt", "w") as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")

print("Results saved to model_results.txt")

# Results dictionary (Modified for CSV)
results = {}

# Evaluate models
for model_name, model in models.items():
    #... (model evaluation code) ...

    # Store results for CSV
    results[model_name] = {
        'CV Results': cv_results.tolist(),
        'Mean Accuracy': cv_results.mean(),
        'Time Taken': elapsed_time
    }

# Convert results to CSV
import pandas as pd    
df_results = pd.DataFrame.from_dict(results, orient='index') 
df_results.to_csv('model_results.csv') 