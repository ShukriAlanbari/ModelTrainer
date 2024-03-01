import main
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


class UserInput:
    def __init__ (self):
        self.data = None
        self.file_path = None
        self.ml_type = None
        self.target_column = None
        

    def run_all(self):
        # self.get_file_path()
        # self.ml_type = self.get_ml_type()
        # self.target_column = self.get_target_column()
        print("1")

    def get_file_path(self):
        while True:
            try:
                file_path = input("Enter the path or name of the CSV file: ")
                self.data = pd.read_csv(file_path)
                self.file_path = file_path
                break
            except FileNotFoundError:
                print("File not found. Please provide a valid file path.")

    def get_ml_type(self):
        while True:
            try:
                ml_type = input("Enter 'regressor' or 'classifier' for the type of machine learning: ").lower()
                if ml_type in ['regressor', 'classifier']:
                    return ml_type
                else:
                    raise ValueError("Invalid input. Please enter either 'regressor' or 'classifier'.")
            except ValueError as e:
                print(e)

    def get_target_column(self):
        print("Available columns in the dataset:")
        print(self.data.columns)
        
        while True:
            try:
                target_column = input("Enter the name of the target column: ") 
                if target_column in self.data.columns:
                    if pd.api.types.is_numeric_dtype(self.data[target_column]):
                        if self.ml_type == 'regressor':
                            return target_column
                            
                        else:
                            print(f"Selected ML type is 'classifier', but target column '{target_column}' is continuous.")
                            
                            
                    elif self.data[target_column].dtype == 'object':
                        if self.ml_type == 'classifier':
                            return target_column
                            
                        else:
                            print(f"Selected ML type is 'regressor', but target column '{target_column}' is categorical.")
                            self.ml_type = self.get_ml_type()

                    else:
                        raise ValueError(f"Unsupported data type for target column '{target_column}'.")
                else:
                    raise ValueError(f"Column '{target_column}' not found. Please choose a valid target column.")
            except ValueError as e:
                print(e)
        

class DataProcessor:
    def __init__(self):
        self.data = None
        self.file_path = None

    @staticmethod
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
                    data = DataProcessor.fill_missing_values(data)
                    print("Missing values filled. Data is now ready for machine learning.")
                    return True, data
                elif user_choice == '2':
                    UserInput().run_all()  # Run UserInput to choose another CSV
                    break
                else:
                    print("Invalid choice. Please enter either 1 or 2.")
        else:
            return True, data

    @staticmethod
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
        return data

    @staticmethod
    def create_dummy_variables(data, categorical_columns):
        print("Creating dummies using OneHotEncoder...")
        encoder = OneHotEncoder(drop='first', sparse=False)
        dummy_variables = encoder.fit_transform(data[categorical_columns])

        data = pd.concat([data, pd.DataFrame(dummy_variables, columns=encoder.get_feature_names_out(categorical_columns))], axis=1)

        print("Dummy variables created successfully.")
        return data

    @staticmethod
    def check_value_types(data):
        categorical_columns = data.select_dtypes(include=['object']).columns

        if not categorical_columns.empty:
            print("Data has categorical or string data and is not ready for machine learning:")
            print("Columns with categorical data:", categorical_columns.tolist())
            print("1: Create Dummy Variables")
            print("2: Choose Another CSV")

            while True:
                user_choice = input("Enter your choice (1 or 2): ")

                if user_choice == '1':
                    return True, DataProcessor.create_dummy_variables(data, categorical_columns)
                elif user_choice == '2':
                    UserInput().run_all()  # Run UserInput to choose another CSV
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            return True, data









user_input_instance = UserInput()
user_input_instance.run_all()
# DataProcessor_instance = DataProcessor()