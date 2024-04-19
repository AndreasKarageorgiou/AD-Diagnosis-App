import pandas as pd

def main():
    # Load the dataset
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

    # Drop additional columns
    columns_to_drop = diagnosis_features + ['APTyear', 'Examyear', 'PTDOBYear', 'DXCURREN']
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Save the updated dataframe to a new CSV file
    df.to_csv("Updated_AIBL.csv", index=False)

    # Get min and max values for each column
    min_values = df.min()
    max_values = df.max()

    # Print min and max values
    print("Minimum values:\n", min_values)
    print("Maximum values:\n", max_values)

    return df

# Call the main function
updated_df = main()
