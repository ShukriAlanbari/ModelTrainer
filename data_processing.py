import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib






def get_file_path():
    while True:
        try:
            file_path = input("Enter the path or name of the CSV file: ")
            data = pd.read_csv(file_path)
            return file_path, data
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")

def get_ml_type():
    while True:
        try:
            ml_type = input("Enter 'regressor' or 'classifier' for the type of machine learning: ").lower()
            if ml_type in ['regressor', 'classifier']:
                return ml_type
            else:
                raise ValueError("Invalid input. Please enter either 'regressor' or 'classifier'.")
        except ValueError as e:
            print(e)

def get_target_column(data, ml_type):
    print("Available columns in the dataset:")
    print(data.columns)
    
    while True:
        try:
            target_column = input("Enter the name of the target column: ") 
            if target_column in data.columns:
                ## Validate if the target column 
                if data[target_column].dtype in ['float64', 'int64']:
                    if ml_type == 'regressor':
                        return target_column
                    else:
                        print(f"Selected ML type is 'classifier', but target column '{target_column}' is continuous.")
                        ml_type = get_ml_type()
                        
                elif data[target_column].dtype == 'object':
                    if ml_type == 'classifier':
                        return target_column
                    else:
                        print(f"Selected ML type is 'regressor', but target column '{target_column}' is categorical.")
                        ml_type = get_ml_type()

                else:
                    raise ValueError(f"Unsupported data type for target column '{target_column}'.")
            else:
                raise ValueError(f"Column '{target_column}' not found. Please choose a valid target column.")
        except ValueError as e:
            print(e)

def check_missing_values(data):
    # Check for missing values
    if data.isnull().any().any():
        print("Data is not ready for machine learning:")
        print("1. Fill Missing Values")
        print("2. Choose Another CSV")

        while True:
            user_choice = input("Enter your choice (1 or 2): ")

            if user_choice == '1':
                # Allow the user to fill in missing values
                data = fill_missing_values(data)
                print("Missing values filled. Data is now ready for machine learning.")
                return True, data
            elif user_choice == '2':
                get_file_path()
                check_missing_values(data)
                break
            else:
                print("Invalid choice. Please enter either 1 or 2.")
    else:
        print("Data is ready for machine learning.")
        return True, data
    
def fill_missing_values(data):
    print("Filling missing data using SimpleImputer...")

    for column in data.columns:
        if data[column].isnull().any():
            print(f"\nColumn: {column}")

            if pd.api.types.is_numeric_dtype(data[column]):
                print("Data type: Numeric")

                while True:
                    fill_strategy = input("Choose a filling strategy (mean, median, custom): ").lower()

                    if fill_strategy == 'custom':
                        custom_value = input("Enter the custom value to fill missing data: ")
                        try:
                            data[column].fillna(float(custom_value), inplace=True)
                            break
                        except ValueError:
                            print("Invalid input. Please enter a valid numeric value.")
                    elif fill_strategy in ['mean', 'median']:
                        imputer = SimpleImputer(strategy=fill_strategy)
                        data[column] = imputer.fit_transform(data[[column]])
                        break
                    else:
                        print("Invalid strategy. Please choose a valid option.")

            else:
                print("Data type: Categorical")

                while True:
                    fill_strategy = input("Choose a filling strategy (mode, custom): ").lower()

                    if fill_strategy == 'custom':
                        custom_value = input("Enter the custom value to fill missing data: ")
                        data[column].fillna(custom_value, inplace=True)
                        break
                    elif fill_strategy == 'mode':
                        imputer = SimpleImputer(strategy=fill_strategy)
                        data[column] = imputer.fit_transform(data[[column]])
                        break
                    else:
                        print("Invalid strategy. Please choose a valid option.")

    print("\nMissing data filled successfully.")

def check_value_types(data):
    categorical_columns = data.select_dtypes(include=['object']).columns

    if not categorical_columns.empty:
        print("Data has categorical or string data and not ready for machine learning:")
        print("Columns with categorical data:", categorical_columns.tolist())
        print("1: Create Dummy Variables")
        print("2: Choose Another CSV")
        while True:
            user_choice = input("Enter your choice (1 or 2): ")

            if user_choice == '1':
                # Create dummy variables
                data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
                print("Dummy variables created successfully.")
                return True, data
            elif user_choice == '2':
                get_file_path()
                return False, None
            else:
                print("Invalid choice. Please enter 1 or 2.")
    else:
        print("No categorical or string data found.")
        return True, data
    
def check_outliers(data,column):
    # Calculate IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    # Print outliers
    print("Outliers:")
    print(outliers)

    # Ask user for choice
    while True:
        user_choice = input("Enter your choice (1: Remove Outliers, 2: Keep Outliers and Continue): ")

        if user_choice == '1':
            # Remove outliers
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            print("Outliers removed. Data is now ready for machine learning.")
            return True, data
        elif user_choice == '2':
            print("Keeping outliers and continuing.")
            return True, data
        else:
            print("Invalid choice. Please enter 1 or 2.")







# Load a sample CSV file with missing values, categorical columns, and outliers
sample_data = pd.DataFrame({
    'Numeric_Column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Categorical_Column': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

# Introduce some missing values
sample_data.loc[[2, 5], 'Numeric_Column'] = None

# Introduce an outlier
sample_data.loc[9, 'Numeric_Column'] = 20

# Display the sample data
print("Sample Data:")
print(sample_data)

# Test the code
file_path, data = get_file_path()

# Choose 'regressor' or 'classifier'
ml_type = get_ml_type()

# Choose the target column
target_column = get_target_column(data, ml_type)

# Check missing values
is_data_ready, data = check_missing_values(data)

# Check value types and process them if needed
is_data_ready, data = check_value_types(data)

# Check outliers for the 'Numeric_Column'
is_data_ready, data = check_outliers(data, 'Numeric_Column')

# Display the final processed data
print("Final Processed Data:")
print(data)
