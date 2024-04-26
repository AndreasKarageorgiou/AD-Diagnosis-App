import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the dataset
    try:
        df = pd.read_csv("AIBL.csv")  # Ensure this path is correct
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: The file could not be found.")
        return

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
            return -1  # Handle cases where none of the conditions are met

    # Apply the function to create a target variable 'DXTYPE'
    df['DXTYPE'] = df.apply(lambda row: create_DXTYPE(row['DXNORM'], row['DXMCI'], row['DXAD']), axis=1)

    # Calculate age before dropping 'Examyear' and 'PTDOBYear'
    if 'Examyear' in df.columns and 'PTDOBYear' in df.columns:
        df["ExamAge"] = df["Examyear"] - df["PTDOBYear"]
    
    # Drop additional columns
    columns_to_drop = diagnosis_features + ['Examyear', 'PTDOBYear', 'DXCURREN']
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Calculate mean and standard deviation for each column
    stats = df.describe().loc[['mean', 'std']]
    print("Mean and standard deviation:\n", stats)

    # Perform Exploratory Data Analysis (EDA)
    # Histograms for each numerical feature
    df.hist(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # Correlation matrix heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix')
    plt.show()

    # Save the updated dataframe to a new CSV file
    df.to_csv("Updated_AIBL.csv", index=False)
    print("Updated dataset saved as 'Updated_AIBL.csv'.")

if __name__ == "__main__":
    main()
