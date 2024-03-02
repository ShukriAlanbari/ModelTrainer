from model_trainer import Regressor, Classifier
import pandas as pd
import numpy as np
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
   
    def __init__(self, data, file_path, ml_type, target_column):
        self.target_column = target_column
        self.data = data
        self.file_path = file_path
        self.ml_type = ml_type
        while True :  
            copy_choice = input("Make a copy of the file?  1.Yes 2.No \n> ").lower()
            print("")
         
            if copy_choice == '1':
                
                self.save_copy()
                
                self.data = pd.read_csv(self.file_path)
                break
            elif copy_choice == '2':
                
                break
            else:
                print("Invalid choice. Please chose 1.Yes or 2.No")

    def run_all(self):
        self.check_missing_values()
        self.check_value_types()
        self.scaler()
        self.data.to_csv(self.file_path, index=False)
        if self.ml_type == "regressor":
            regressor_instance = Regressor(self.data, self.target_column)
            regressor_instance.run_all()
        elif self.ml_type == "classifier":
            classifier_instance = Classifier(self.data, self.target_column)
            classifier_instance.run_all()
    
    def check_missing_values(self):
        print("")
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
                    self.run_all() 
                    
                else:
                    print("Invalid choice. Please enter either 1 or 2.")
        else:
            return True
    
    def fill_missing_values(self):
                
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

    def check_value_types(self):
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns

        if not self.categorical_columns.empty:
            print("Data has categorical or string data and is not ready for machine learning:")
            print("Columns with categorical data:", self.categorical_columns.tolist())
            print("1: Create Dummy Variables")
            print("2: Choose Another CSV")

            while True:
                user_choice = input("Enter your choice (1 or 2): ")

                if user_choice == '1':
                    self.create_dummy_variables(self.categorical_columns)
                    return True
                    
                elif user_choice == '2':
                    user_input_instance = UserInput()  
                    user_input_instance.run_all()
                    self.run_all()
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            return True

    def create_dummy_variables(self, categorical_columns):
        print("Creating dummies using pd.get_dummies...")
        
        
        for column in categorical_columns:
            dummy_variables = pd.get_dummies(self.data[column], prefix=column, drop_first=True)
            
            
            self.data = pd.concat([self.data.drop(column, axis=1), dummy_variables], axis=1)
        
        print("Dummy variables created successfully.")
        return self.data
    
    def save_copy(self):
        if '\\' in self.file_path:
            directory, filename = self.file_path.rsplit('\\', 1)
        else:
            directory = ''
            filename = self.file_path

        new_filename = filename.replace(".", "(M).")
        new_file_path = f"{directory}\\{new_filename}"

        shutil.copyfile(self.file_path, new_file_path)
        self.file_path = new_file_path
        print(f"Data copied to: {new_file_path}")
        
   
    def scaler(self):
        if self.ml_type == "regressor":
            scaling_choice= input("Do you want to scale the data?\ 1.Yes     2.No\n> ")
            if scaling_choice == "1":
                numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

                if not numeric_columns.empty:
                    print("Scaling numeric features using StandardScaler...")
                    scaler = StandardScaler()

                    self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])

                    print("Numeric features scaled successfully.\n")
                    return self.data
                else:
                    print("No numeric features to scale.")
                    return self.data
            elif scaling_choice == "2":
                pass
            else:
                print("Invalid choice.Please choose 1.Yes  2.No")
        else:
            pass


