import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("AIBL.csv")

# Define diagnosis features
diagnosis_features = ['DXNORM', 'DXMCI', 'DXAD']

# Function to create DXTYPE
def create_DXTYPE(DXNORM, DXMCI, DXAD):
    if DXNORM == 1:
        return 0   
    elif DXMCI == 1:
        return 1  
    elif DXAD == 1:
        return 2  
    else:
        return -1  

# Create DXPOS before removing other diagnosis features
df['DXTYPE'] = df.apply(lambda row: create_DXTYPE(row['DXNORM'], row['DXMCI'], row['DXAD']), axis=1)

# Get value counts for 'DXTYPE'
dxtypes_counts = df['DXTYPE'].value_counts()

# Print the result
print("Value Counts for DXTYPE:")
print(dxtypes_counts)

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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3)),
    "HistGradientBoosting": HistGradientBoostingClassifier(), # Relatively new model
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Initialize lists to store results
mean_accuracies = []
std_deviations = []
cv_times = []

# Perform 10-fold cross-validation for each classifier
for name, clf in classifiers.items():
    start_time = time.time()
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=10)
    end_time = time.time()
    
    # Fit the model to get predictions for classification report
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate classification report metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    mean_accuracies.append(scores.mean())
    std_deviations.append(scores.std())
    cv_times.append(end_time - start_time)

    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    print("=" * 50)

# Create DataFrame from results
results_df = pd.DataFrame({
    "Classifier": list(classifiers.keys()),
    "Mean Accuracy": mean_accuracies,
    "Standard Deviation": std_deviations,
    "Cross-Validation Time": cv_times
})

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Classifier', y='Mean Accuracy', hue='Classifier', palette='viridis', legend=False)
plt.title('Mean Accuracy of Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Mean Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
