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
import joblib






def get_user_input():
    print("Welcome to the ML Model Selector App!")

    ## Get the File Path ##
    while True:
        try:
            file_path = input("Enter the path or name of the CSV file: ")
            data = pd.read_csv(file_path)
            break
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")

    ## Get The ML Type ##
    while True:
        try:
            ml_type = input("Enter 'regressor' or 'classifier' for the type of machine learning: ").lower()
            if ml_type in ['regressor', 'classifier']:
                break
            else:
                raise ValueError("Invalid input. Please enter either 'regressor' or 'classifier'.")
        except ValueError as e:
            print(e)

    ## Get the Target Column ##
    print("Available columns in the dataset:")
    print(data.columns)
    
    while True:
        try:
            target_column = input("Enter the name of the target column: ") 
            if target_column in data.columns:
                break
            else:
                raise ValueError(f"Column '{target_column}' not found. Please choose a valid target column.")
        except ValueError as e:
            print(e)

    return ml_type, file_path, target_column




if __name__ == "__main__":
    get_user_input()