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
import shutil
import os



class UserInput:
    def __init__ (self):
        self.data = None
        self.file_path = None
        self.ml_type = None
        self.target_column = None
        
    def run_all(self):
        self.get_file_path()
        self.ml_type = self.get_ml_type()
        self.target_column = self.get_target_column()
        
    def get_file_path(self):
        while True:
            try:
                file_path = input("Enter the path or name of the CSV file: ")
                print("")
                self.data = pd.read_csv(file_path)
                self.file_path = file_path
                break
            except FileNotFoundError:
                print("File not found. Please provide a valid file path.\n")

    def get_ml_type(self):
        while True:
            try:
                ml_type = input("Enter 'regressor' or 'classifier' for the type of machine learning: ").lower()
                if ml_type in ['regressor', 'classifier']:
                    return ml_type
                else:
                    raise ValueError("Invalid input. Please enter either 'regressor' or 'classifier'.\n")
            except ValueError as e:
                print(e)

    def get_target_column(self):
        print("\nAvailable columns in the dataset:")
        print(self.data.columns)
        
        while True:
            try:
                target_column = input("Enter the name of the target column: ")
                print("")
                if target_column in self.data.columns:
                    if pd.api.types.is_numeric_dtype(self.data[target_column]):
                        if self.ml_type == 'regressor':
                            return target_column
                            
                        else:
                            print(f"Selected ML type is 'classifier', but target column '{target_column}' is continuous.(Regressor is recommended)")
                            
                            
                    elif self.data[target_column].dtype == 'object':
                        if self.ml_type == 'classifier':
                            return target_column
                            
                        else:
                            print(f"Selected ML type is 'regressor', but target column '{target_column}' is categorical. (Classifier is recommended)")
                            self.ml_type = self.get_ml_type()

                    else:
                        raise ValueError(f"Unsupported data type for target column '{target_column}'.\n")
                else:
                    raise ValueError(f"Column '{target_column}' not found. Please choose a valid target column.\n")
            except ValueError as e:
                print(e)
        

class DataProcessor:
    def __init__(self, data, file_path):
        self.data = data
        self.file_path = file_path
        
    def run_all(self):
        self.check_missing_values()
        self.check_value_types()

    
    def check_missing_values(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        # Check for missing values
        if self.data.isnull().any().any():
            print("Data is not ready for machine learning:")
            print("1. Fill Missing Values")
            print("2. Choose Another CSV")

            while True:
                user_choice = input("Enter your choice (1 or 2): ")

                if user_choice == '1':
                    
                    self.data = self.fill_missing_values()
                    return True
                elif user_choice == '2':
                    user_input_instance = UserInput()  
                    user_input_instance.run_all()
                    data_processor_instance.run_all() 
                    
                else:
                    print("Invalid choice. Please enter either 1 or 2.")
        else:
            return True

    
    def fill_missing_values(self):
        copy_choice = input("Make a copy of the file?  1. Yes 2. No \n> ").lower()
        print("")
        while True: 
            if copy_choice == '1':
                # Save a copy of the original data
                self.save_copy()
                # Read the data from the copy for filling missing values
                self.data = pd.read_csv(self.file_path.replace(".", "(M)."))
                break
            elif copy_choice == '2':
                # Continue working with the original data
                break
            else:
                print("Invalid choice. Please chose 1 > Yes or 2 > No")

        
        print("Filling missing data using SimpleImputer...")

        for column in self.data.columns:
            if self.data[column].isnull().any():
                print(f"\nColumn: {column}")

                if pd.api.types.is_numeric_dtype(self.data[column]):
                    print("Data type: Numeric")

                    while True:
                        fill_strategy = input("Choose a filling strategy (mean, median, custom): ").lower()

                        if fill_strategy == "custom":
                            custom_value = input("Enter the custom value to fill missing data: ")
                            try:
                                self.data[column].fillna(float(custom_value), inplace=True)
                                break
                            except ValueError:
                                print("Invalid input. Please enter a valid numeric value.")
                        elif fill_strategy in ["mean", "median"]:
                            imputer = SimpleImputer(strategy=fill_strategy)
                            self.data[column] = imputer.fit_transform(self.data[[column]])
                            break
                        else:
                            print("Invalid strategy. Please choose a valid option.")

                else:
                    print("Data type: Categorical")

                    while True:
                        fill_strategy = input("Choose a filling strategy (1. Most Frequent / 2. custom): ").lower()

                        if fill_strategy == "2":
                            custom_value = input("Enter the custom value to fill missing data: ")
                            self.data[column].fillna(custom_value, inplace=True)
                            break
                        elif fill_strategy == "1":
                            imputer = SimpleImputer(strategy="most_frequent")
                            self.data[column] = imputer.fit_transform(self.data[[column]]).ravel()
                            break
                        else:
                            print("Invalid strategy. Please choose a valid option.")

        print("\nMissing data filled successfully.")
        return self.data

   
    def create_dummy_variables(self, categorical_columns):
        print("Creating dummies using OneHotEncoder...")
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        dummy_variables = encoder.fit_transform(self.data[categorical_columns])

        self.data = pd.concat([self.data, pd.DataFrame(dummy_variables, columns=encoder.get_feature_names_out(categorical_columns))], axis=1)

        print("Dummy variables created successfully.")
        return self.data

    
    def check_value_types(self):
        categorical_columns = self.data.select_dtypes(include=['object']).columns

        if not categorical_columns.empty:
            print("Data has categorical or string data and is not ready for machine learning:")
            print("Columns with categorical data:", categorical_columns.tolist())
            print("1: Create Dummy Variables")
            print("2: Choose Another CSV")

            while True:
                user_choice = input("Enter your choice (1 or 2): ")

                if user_choice == '1':
                    self.create_dummy_variables(categorical_columns)
                    return True
                    
                elif user_choice == '2':
                    user_input_instance = UserInput()  
                    user_input_instance.run_all()
                    data_processor_instance.run_all()
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            return True

    def save_copy(self):
        if '\\' in self.file_path:
            directory, filename = self.file_path.rsplit('\\', 1)
        else:
            directory = ''
            filename = self.file_path

        new_filename = filename.replace(".", "(M).")
        new_file_path = f"{directory}\\{new_filename}"

        shutil.copyfile(self.file_path, new_file_path)
        print(f"Data copied to: {new_file_path}")
   
   






user_input_instance = UserInput()
user_input_instance.run_all()
data_processor_instance = DataProcessor(user_input_instance.data.copy(), user_input_instance.file_path)
data_processor_instance.run_all()

# Print the head of both the original and modified data to check the changes
print("Original Data:")
print(user_input_instance.data.head())

print("\nModified Data:")
print(data_processor_instance.data.head())