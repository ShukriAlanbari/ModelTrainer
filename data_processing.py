from model_trainer import Regressor, Classifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import shutil
import os


# Class to handle user input for model training.
class UserInput:

    
    def __init__ (self):
        self.data = None
        self.file_path = None
        self.ml_type = None
        self.target_column = None
    
    """Execute the sequence of methods to gather user input."""
    def run_all(self):
        # Execute the sequence of methods to gather user input
        self.get_file_path()
        self.ml_type = self.get_ml_type()
        self.target_column = self.get_target_column()

    """Prompt the user for a valid file path and load CSV data."""   
    def get_file_path(self):
        # Continuously prompt the user until a valid file path is provided
        while True:
            try:
                file_path = input("Enter the path or name of the CSV file: ")
                print("")
                self.data = pd.read_csv(file_path) # Read CSV data from the provided file path
                self.file_path = file_path # Set the instance variable with the provided file path
                break   # Exit the loop if successful
            except FileNotFoundError:
                print("File not found. Please provide a valid file path.\n") # Inform the user of the error

    """Prompt the user for the type of machine learning: 'regressor' or 'classifier'."""
    def get_ml_type(self):
        # Continuously prompt the user until a valid machine learning type is provided
        while True:
            try:
                # Prompt user for ML type
                ml_type = input("Enter 'regressor' or 'classifier' for the type of machine learning: ").lower()
                if ml_type in ['regressor', 'classifier']:
                    return ml_type # Return the valid ML type if provided
                else:
                    # Inform the user of the error
                    raise ValueError("Invalid input. Please enter either 'regressor' or 'classifier'.\n")
            except ValueError as e:
                print(e) # Print the error message if an exception occurs

    """Prompt the user for the target column and validate its compatibility with the selected ML type."""
    def get_target_column(self):
        # Display available columns in the loaded dataset
        print("\nAvailable columns in the dataset:")
        print(self.data.columns)
        
        while True:
            try:
                target_column = input("Enter the name of the target column: ")
                print("")

                # Check if the entered target column exists in the dataset
                if target_column in self.data.columns:

                    # Check if the target column is of numeric data type
                    if pd.api.types.is_numeric_dtype(self.data[target_column]):

                        # Check the ML type to ensure compatibility with a numeric target column
                        if self.ml_type == 'regressor':
                            return target_column # Return the valid target column for regression
                            
                        else:
                            # Inform the user about the mismatched ML type and recommend a change
                            print(f"Selected ML type is 'classifier', but target column '{target_column}' is continuous.(Regressor is recommended)")
                            
                            
                    elif self.data[target_column].dtype == 'object':

                        # Check the ML type to ensure compatibility with a categorical target column
                        if self.ml_type == 'classifier':
                            return target_column # Return the valid target column for classification
                            
                        else:
                            # Inform the user about the mismatched ML type and recommend a change
                            print(f"Selected ML type is 'regressor', but target column '{target_column}' is categorical. (Classifier is recommended)")
                            self.ml_type = self.get_ml_type() # Prompt the user to re-enter the ML type

                    else:
                        # Raise an error for unsupported data types in the target column
                        raise ValueError(f"Unsupported data type for target column '{target_column}'.\n")
                else:
                    # Raise an error if the entered target column is not found in the dataset
                    raise ValueError(f"Column '{target_column}' not found. Please choose a valid target column.\n")
            except ValueError as e:
                print(e) # Print the error message if an exception occurs
        

# Class to process data for machine learning.
class DataProcessor:
    
    """Initialize instance variables."""
    def __init__(self, data, file_path, ml_type, target_column):
        """
        Initialize the instance variables with the provided parameters.

        Parameters:
        - data: pandas DataFrame, the input dataset
        - file_path: str, the file path of the input dataset
        - ml_type: str, the type of machine learning task ('regressor' or 'classifier')
        - target_column: str, the target column for machine learning
        """

        self.target_column = target_column
        self.data = data
        self.file_path = file_path
        self.ml_type = ml_type

        # Prompt the user to make a copy of the file
        while True :  
            copy_choice = input("Make a copy of the file?  1.Yes 2.No \n> ").lower()
            print("")

            # Check user's choice
            if copy_choice == '1':
                
                # Call the method to save a copy of the file
                self.save_copy()
                
                # Reload the data from the copied file
                self.data = pd.read_csv(self.file_path)
                break   # Exit the loop if successful
            elif copy_choice == '2':
                
                break   # Exit the loop if user chooses not to make a copy
            else:
                print("Invalid choice. Please chose 1.Yes or 2.No") # Inform the user about the invalid choice

    """Run all data processing steps."""
    def run_all(self):

        # Check for missing values in the dataset
        self.check_missing_values()
        # Check and validate the data types of columns
        self.check_value_types()
        # Standardize or normalize the data using a scaler
        self.scaler()
        # Save the processed data back to the original file path
        self.data.to_csv(self.file_path, index=False)

        # Choose the appropriate machine learning task based on the provided ML type
        if self.ml_type == "regressor":
            # Create an instance of the Regressor class and run its methods
            regressor_instance = Regressor(self.data, self.target_column, self.file_path)
            regressor_instance.run_all()
        elif self.ml_type == "classifier":
            # Create an instance of the Classifier class and run its methods
            classifier_instance = Classifier(self.data, self.target_column, self.file_path)
            classifier_instance.run_all()

    """Check for missing values in the dataset."""      
    def check_missing_values(self):
        """
        Check for missing values in the dataset.

        Returns:
        - bool: True if missing values have been handled, False otherwise.
        """
        print("")
        # Check for missing values
        if self.data.isnull().any().any():
            print("Data is not ready for machine learning:")
            print("1. Fill Missing Values")
            print("2. Choose Another CSV")

            while True:
                user_choice = input("Enter your choice (1 or 2): ")

                if user_choice == '1':
                    # Call the method to fill missing values and update the dataset
                    self.data = self.fill_missing_values()
                    return True # Indicate that missing values have been handled
                elif user_choice == '2':
                    # Create a new instance of UserInput and run the data input process again
                    user_input_instance = UserInput()  
                    user_input_instance.run_all()
                    self.run_all() # Rerun the entire process with the new input
                    
                else:
                    print("Invalid choice. Please enter either 1 or 2.")
        else:
            return True  # Indicate that there are no missing values in the dataset
    
    """Fill missing values in the dataset."""
    def fill_missing_values(self): 
        """
        Fill missing values in the dataset.

        Returns:
        - pd.DataFrame: The updated dataset after filling missing values.
        """

        # Inform the user about the process of filling missing data     
        print("Filling missing data using SimpleImputer...")

        # Iterate through each column in the dataset
        for column in self.data.columns:
            # Check if the column has missing values
            if self.data[column].isnull().any():
                print(f"\nColumn: {column}")

                # Check if the column has numeric data type
                if pd.api.types.is_numeric_dtype(self.data[column]):
                    print("Data type: Numeric")

                    while True:
                        # Prompt the user to choose a filling strategy for numeric columns
                        fill_strategy = input("Choose a filling strategy (mean, median, custom): ").lower()

                        if fill_strategy == "custom":
                            # If the user chooses a custom strategy, prompt for a custom value
                            custom_value = input("Enter the custom value to fill missing data: ")
                            try:
                                # Try to fill missing values with the provided custom value
                                self.data[column].fillna(float(custom_value), inplace=True)
                                break
                            except ValueError:
                                print("Invalid input. Please enter a valid numeric value.")
                        elif fill_strategy in ["mean", "median"]:
                            # If the user chooses mean or median, use SimpleImputer with the selected strategy
                            imputer = SimpleImputer(strategy=fill_strategy)
                            self.data[column] = imputer.fit_transform(self.data[[column]])
                            break
                        else:
                            print("Invalid strategy. Please choose a valid option.")

                else:
                    print("Data type: Categorical")

                    while True:
                        # Prompt the user to choose a filling strategy for categorical columns
                        fill_strategy = input("Choose a filling strategy (1. Most Frequent / 2. custom): ").lower()

                        if fill_strategy == "2":
                            # If the user chooses a custom strategy, prompt for a custom value
                            custom_value = input("Enter the custom value to fill missing data: ")
                            self.data[column].fillna(custom_value, inplace=True)
                            break
                        elif fill_strategy == "1":
                            # If the user chooses most frequent, use SimpleImputer with the most_frequent strategy
                            imputer = SimpleImputer(strategy="most_frequent")
                            self.data[column] = imputer.fit_transform(self.data[[column]]).ravel()
                            break
                        else:
                            print("Invalid strategy. Please choose a valid option.")

        # Inform the user that missing data has been filled successfully
        print("Missing data filled successfully.\n")
        return self.data    # Return the updated dataset
    
    """Check and validate the data types of columns."""
    def check_value_types(self):
        """
        Check and validate the data types of columns.

        Returns:
        - bool: True if categorical columns have been processed, False otherwise.
        """

        # Identify columns with categorical data
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns

        # Check if there are any categorical columns in the dataset
        if not self.categorical_columns.empty:
            # Inform the user about the presence of categorical or string data
            print("Data has categorical or string data and is not ready for machine learning:")
            print("Columns with categorical data:", self.categorical_columns.tolist())
            print("1: Create Dummy Variables")
            print("2: Choose Another CSV")

            # Prompt the user for action based on categorical data
            while True:
                user_choice = input("Enter your choice (1 or 2): ")

                # Execute the chosen action
                if user_choice == '1':
                    # Call the method to create dummy variables for categorical columns
                    self.create_dummy_variables(self.categorical_columns)
                    return True     # Indicate that categorical columns have been processed
                    
                elif user_choice == '2':
                    # Create a new instance of UserInput and run the data input process again
                    user_input_instance = UserInput()  
                    user_input_instance.run_all()
                    self.run_all()   # Rerun the entire process with the new input
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            return True # Indicate that there are no categorical columns in the dataset

    """Create dummy variables for categorical columns."""
    def create_dummy_variables(self, categorical_columns):
        """
        Create dummy variables for categorical columns.

        Parameters:
        - categorical_columns: list, the list of categorical columns

        Returns:
        - pd.DataFrame: The updated dataset after creating dummy variables.
        """

        # Inform the user about the process of creating dummy variables
        print("Creating dummies using pd.get_dummies...")

        # Initialize a list to store dummy column names related to the target column
        target_dummy_columns = []  

        # Iterate through each categorical column
        for column in categorical_columns:
            # Check if the current column is the target column
            if column == self.target_column:
                # Handle the target column differently by creating dummy variables with a modified prefix
                dummy_variables = pd.get_dummies(self.data[column], prefix=f"{column}_dummy", drop_first=True)
                # Extend the list of target-related dummy columns
                target_dummy_columns.extend(dummy_variables.columns)
                
                # Concatenate the dummy variables with the dataset and drop the original target column
                self.data = pd.concat([self.data, dummy_variables], axis=1)
                self.data = self.data.drop(column, axis=1)
            else:
                # Encode non-target categorical columns using pd.get_dummies
                dummy_variables = pd.get_dummies(self.data[column], prefix=column, drop_first=True)

                # Concatenate the dummy variables with the dataset and drop the original categorical column
                self.data = pd.concat([self.data, dummy_variables], axis=1)
                self.data = self.data.drop(column, axis=1)

        # Update the target_column if dummy variables were created for it
        if target_dummy_columns:
            self.target_column = target_dummy_columns[0]
        
        # Inform the user that dummy variables have been created successfully
        print("Dummy variables created successfully.\n")
        return self.data    # Return the updated dataset
    
    """Save a copy of the original file."""
    def save_copy(self):
        """
        Save a copy of the original file.

        Creates a new file with the same content as the original file.

        Returns:
        - None
        """

        # Check if the file path contains a directory structure
        if '\\' in self.file_path:
            # If yes, split the file path into directory and filenam
            directory, filename = self.file_path.rsplit('\\', 1)
        else:
            # If no directory structure, set the directory as an empty string-
            # and the filename as the entire file path
            directory = ''
            filename = self.file_path

         # Create a new filename by replacing the first dot (.) with (M).
        new_filename = filename.replace(".", "(M).")
        # Create the new file path by combining the directory and the new filename
        new_file_path = f"{directory}\\{new_filename}"

        # Copy the original file to the new file path
        shutil.copyfile(self.file_path, new_file_path)
        # Update the instance variable with the new file path
        self.file_path = new_file_path
        # Inform the user about the successful copy operation
        print(f"Data copied to: {new_file_path}")

    """Scale numeric features using StandardScaler if the machine learning type is "regressor"."""
    def scaler(self):
        """
        Scale numeric features using StandardScaler if the machine learning type is "regressor".

        This method prompts the user for scaling preferences and applies StandardScaler to the numeric features
        if the user chooses to scale. It then informs the user about the success of the scaling operation.

        Returns:
        - pd.DataFrame: The updated dataset after scaling numeric features if applicable, or the original dataset.
        """
        
        # Check if the ML type is "regressor"
        if self.ml_type == "regressor":
            # Prompt the user for scaling choice
            scaling_choice= input("Do you want to scale the data?\ 1.Yes     2.No\n> ")

            # Process the user's choice for scaling
            if scaling_choice == "1":
                # Identify numeric columns for scaling
                numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

                # Check if there are numeric columns to scale
                if not numeric_columns.empty:
                    # Inform the user about scaling numeric features using StandardScaler
                    print("Scaling numeric features using StandardScaler...")
                    # Initialize the StandardScaler
                    scaler = StandardScaler()

                    # Apply StandardScaler to the numeric columns
                    self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])

                    # Inform the user about successful scaling
                    print("Numeric features scaled successfully.\n")
                    return self.data    # Return the updated dataset
                else:
                    # Inform the user if there are no numeric features to scale
                    print("No numeric features to scale.")
                    return self.data    # Return the dataset without scaling
            elif scaling_choice == "2":
                # Do nothing if the user chooses not to scale
                pass
            else:
                # Inform the user about an invalid choice
                print("Invalid choice.Please choose 1.Yes  2.No")
        else:
            # Do nothing if the ML type is not "regressor"
            pass


