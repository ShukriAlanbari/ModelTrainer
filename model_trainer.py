import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from joblib import dump
import shutil
import os
import time
warnings.filterwarnings("ignore")



# Regressor class for handling regression tasks with machine learning models.
class Regressor:
    """
    Regressor class for handling regression tasks with machine learning models.

    Attributes:
    - data (pandas.DataFrame): The input dataset.
    - target_column (str or list): The column(s) to be treated as the target variable(s).
      If a list, specified columns will be dropped from the dataset.
      If a string, the single specified column will be dropped.
    - file_path (str): The file path associated with the dataset.
    - X (pandas.DataFrame): The feature matrix after dropping target_column(s).
    - y (pandas.Series): The target variable(s).
    - X_train, X_test, y_train, y_test: Training and testing sets split from the dataset.
    - model_dict (dict): Dictionary to store model-related information.
        Keys:
            - "Regressor": List to store regressor instances.
            - "Best_Parameters": List to store the best parameters for each regressor.
            - "MAE": List to store Mean Absolute Error for each regressor.
            - "RMSE": List to store Root Mean Squared Error for each regressor.
            - "R2_Score": List to store R-squared score for each regressor.
            - "Training_Time": List to store training time for each regressor.
    - best_regressor_data: Variable to store the best regressor data.

    Methods:
    - __init__(self, data, target_column, file_path):
        Initializes the Regressor class with provided data, target column, and file path.

    - run_all(self):
        Run multiple regression models and compare their performance.

    - linear_model(self):
        Train a Linear Regression model with hyperparameter tuning.

    - polynomial_model(self):
        Train a Polynomial Regression model with hyperparameter tuning.

    - lasso_model(self):
        Train a Lasso Regression model with hyperparameter tuning.

    - ridge_model(self):
        Train a Ridge Regression model with hyperparameter tuning.

    - elastic_model(self):
        Train an ElasticNet Regression model with hyperparameter tuning.

    - svr_model(self):
        Train a Support Vector Regressor (SVR) model with hyperparameter tuning.

    - knn_regressor_model(self):
        Train a K-Nearest Neighbors (KNN) Regressor model with hyperparameter tuning.

    - tree_regressor_model(self):
        Train a Decision Tree Regressor model with hyperparameter tuning.

    - forest_regressor_model(self):
        Train a Random Forest Regressor model with hyperparameter tuning.

    - run_regressor(self, model, param_grid, X_train, y_train, X_test, y_test, cv=10):
        Perform hyperparameter tuning, training, and evaluation for a given regression model.

    - compare_regressor(self):
        Compare the performance of different regressors and identify the best-performing regressor.

    - save_regressor(self):
        Asks the user if they want to save the recommended regression model and saves it if desired.
    """
    
    # Initialize an instance of Regressor.
    def __init__(self, data, target_column, file_path) -> None:
        """
        Initializes the Classifier class with provided data, target column, and file path.
        Parameters:
        - data (pandas.DataFrame): The input dataset.
        - target_column (str or list): The column(s) to be treated as the target variable(s).
          If a list, specified columns will be dropped from the dataset.
          If a string, the single specified column will be dropped.
        - file_path (str): The file path associated with the dataset.

        Returns:
        None
        """

        
        # Initialize instance variables with provided parameters
        self.data= data
        self.file_path = file_path
        
        # Check if the target_column is a list
        if isinstance(target_column, list):
            # If yes, drop the specified columns from the dataset
            self.X = self.data.drop(columns=target_column)
        else:
            # If no, drop the single specified column from the dataset
            self.X = self.data.drop(columns=[target_column])
        # Set y as the target_column in the dataset
        self.y = self.data[target_column]
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size= 0.3, random_state= 101)
        
        # Initialize a dictionary to store model-related information
        self.model_dict = { 
                "Regressor": [],
                "Best_Parameters": [] ,
                "MAE": [],
                "RMSE": [],
                "R2_Score": [],
                "Training_Time": [],
                }
        # Initialize a variable to store the best regressor data
        self.best_regressor_data = None
    
    # Run multiple regression models and compare their performance.
    def run_all(self):
        """
        Run multiple regression models and compare their performance.

        The method runs the following regression models:
        - Linear Regression
        - Polynomial Regression
        - Lasso Regression
        - Ridge Regression
        - Elastic Net Regression
        - Support Vector Regressor
        - k-Nearest Neighbors Regressor
        - Decision Tree Regressor
        - Random Forest Regressor

        After running the models, it compares their performance and saves the best-performing regressor.
        """

        # Run linear regression model
        self.linear_model()
        # Run polynomial regression model
        self.polynomial_model()
        # Run Lasso regression model
        self.lasso_model()
        # Run Ridge regression model
        self.ridge_model()
        # Run Elastic Net regression model
        self.elastic_model()
        # Run Support Vector Regressor model
        self.svr_model()
        # Run k-Nearest Neighbors Regressor model
        self.knn_regressor_model()
        # Run Decision Tree Regressor model
        self.tree_regressor_model()
        # Run Random Forest Regressor model
        self.forest_regressor_model()
        # Compare performance of all regressors
        self.compare_regressor()
        # Save the best-performing regressor
        self.save_regressor()

    # Train a Linear Regression model with hyperparameter tuning.
    def linear_model(self):
        """
        Train a Linear Regression model with hyperparameter tuning.

        The method initializes a Linear Regression model and tunes hyperparameters using a predefined parameter grid.
        It then runs the regressor with hyperparameter tuning on the training and testing sets.
        """

        # Inform the user about training the Linear Regression model
        print("Training Linear Regression Model, please wait....")
        
        # Initialize a Linear Regression model
        lr_model = LinearRegression()
        
        # Define a parameter grid for hyperparameter tuning
        lr_param_grid = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],  
        }

        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(lr_model, lr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a Polynomial Regression model with hyperparameter tuning.
    def polynomial_model(self):
        """
        Train a Polynomial Regression model with hyperparameter tuning.

        The method informs the user about training the Polynomial Regression model and defines a parameter grid
        for hyperparameter tuning in Polynomial Regression. It creates a pipeline for Polynomial Regression with
        PolynomialFeatures and Linear Regression and runs the regressor with hyperparameter tuning on the training
        and testing sets.
        """

        # Inform the user about training the Polynomial Regression model
        print("Training Polynomial Regression Model, please wait....")
        
        # Define a parameter grid for hyperparameter tuning in Polynomial Regression
        poly_param_grid = {
            "polynomialfeatures__degree": list(range(1, 11)),
            "polynomialfeatures__include_bias": [True, False],
            "polynomialfeatures__interaction_only": [True, False]
        }
        
        # Create a pipeline for Polynomial Regression with PolynomialFeatures and Linear Regression
        polynomial_model_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(polynomial_model_pipe, poly_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a Lasso Regression model with hyperparameter tuning.
    def lasso_model(self):
        """
        Train a Lasso Regression model with hyperparameter tuning.

        The method informs the user about training the Lasso Regression model and initializes a Lasso Regression model.
        It defines a range of alpha values for regularization and a parameter grid for hyperparameter tuning in Lasso Regression.
        The regressor is then run with hyperparameter tuning on the training and testing sets.
        """
        # Inform the user about training the Lasso Regression model
        print("Training Lasso Model, please wait....")
        
        # Initialize a Lasso Regression model
        lasso_model = Lasso()

        # Define a range of alpha values for regularization
        alphas = np.logspace(-6, 4, 11)  

        # Define a parameter grid for hyperparameter tuning in Lasso Regression
        lasso_param_grid = {
            "alpha": alphas,     # Regularization strength parameter
            "fit_intercept": [True, False],   # Whether to calculate the intercept
            "precompute": [True, False],    # Whether to use precomputed Gram matrix
            "positive": [True, False],  # Constrain the coefficients to be positive
            "selection": ['cyclic', 'random'],  # Strategy for coefficient updates
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000]   # Maximum number of iterations
              
        }
        
        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(lasso_model, lasso_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
    
    # Train a Ridge Regression model with hyperparameter tuning.
    def ridge_model(self):
        """
        Train a Ridge Regression model with hyperparameter tuning.

        The method informs the user about training the Ridge Regression model and initializes a Ridge Regression model.
        It defines a range of alpha values for regularization and a parameter grid for hyperparameter tuning in Ridge Regression.
        The regressor is then run with hyperparameter tuning on the training and testing sets.
        """
        
        # Inform the user about training the Ridge Regression model
        print("Training Ridge Model, please wait....")
        
        # Initialize a Ridge Regression model
        ridge_model = Ridge()

        # Define a range of alpha values for regularization
        alphas = np.logspace(-6, 4, 11)

        # Define a parameter grid for hyperparameter tuning in Ridge Regression
        ridge_param_grid = {
            "alpha": alphas,    # Regularization strength parameter
            "fit_intercept": [True, False],    # Whether to calculate the intercept
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],  # Solver for optimization
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000],  # Maximum number of iterations
        }
        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(ridge_model, ridge_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train an ElasticNet Regression model with hyperparameter tuning.
    def elastic_model(self):
        """
        Train an ElasticNet Regression model with hyperparameter tuning.

        The method informs the user about training the ElasticNet Regression model and initializes an ElasticNet Regression model.
        It defines a range of alpha values for regularization, a range of l1_ratio values for the ElasticNet mixing parameter,
        and a parameter grid for hyperparameter tuning in ElasticNet Regression. The regressor is then run with hyperparameter
        tuning on the training and testing sets.
        """
         
        # Inform the user about training the ElasticNet Model
        print("Training ElasticNet Model, please wait....")
        
        # Initialize an ElasticNet Regression model
        elastic_model = ElasticNet()

        # Define a range of alpha values for regularization
        alphas = np.logspace(-4, 2, 7)
        # Define a range of l1_ratio values for ElasticNet mixing parameter
        l1_ratios = np.linspace(0, 1, num=100) 

        # Define a parameter grid for hyperparameter tuning in ElasticNet Regression
        elastic_param_grid = {
            "alpha": alphas,    # Regularization strength parameter
            "l1_ratio": l1_ratios,  # Mixing parameter between L1 and L2 penalties
            "fit_intercept": [True, False], # Whether to calculate the intercept
            "selection": ['cyclic', 'random'],  # Strategy for coefficient updates
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000], # Maximum number of iterations
            "tol": [1e-4, 1e-3, 1e-2],   # Tolerance for stopping criterion
            "positive": [True, False], # Constrain the coefficients to be positive
        }

        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(elastic_model, elastic_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
    
    # Train a Support Vector Regressor (SVR) model with hyperparameter tuning.
    def svr_model(self):
        """
        Train a Support Vector Regressor (SVR) model with hyperparameter tuning.

        The method informs the user about training the Support Vector Regressor (SVR) model and initializes an SVR model.
        It defines a parameter grid for hyperparameter tuning in SVR and runs the regressor with hyperparameter tuning on
        the training and testing sets.
        """

        # Inform the user about training the Support Vector Regressor (SVR) Model
        print("Training SVR Model, please wait....")
        
        # Initialize a Support Vector Regressor model
        svr_model = SVR()
        
        # Define a parameter grid for hyperparameter tuning in SVR
        svr_param_grid = {
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],  # Type of kernel function
        "C": [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100],  # Regularization parameter
        "epsilon": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0],  # Epsilon parameter in the SVR model
        "gamma": ["auto", "scale"],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        "degree": list(range(2, 6)),  # Degree of the polynomial kernel function
        "coef0": [0.0, 0.1, 0.5, 1.0],  # Independent term in the kernel function
        "shrinking": [True, False],  # Whether to use the shrinking heuristic
        "tol": [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criterion
        "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000],  # Maximum number of iterations
    }
        
        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(svr_model, svr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a K-Nearest Neighbors (KNN) Regressor model with hyperparameter tuning.
    def knn_regressor_model(self):
        """
        Train a K-Nearest Neighbors (KNN) Regressor model with hyperparameter tuning.

        The method informs the user about training the K-Nearest Neighbors (KNN) Regressor model and initializes a KNN Regressor model.
        It defines a parameter grid for hyperparameter tuning in KNN Regressor and runs the regressor with hyperparameter tuning on
        the training and testing sets.
        """

        # Inform the user about training the KNN Regressor Model
        print("Training KNN Regressor Model, please wait....")
        
        # Initialize a KNN Regressor model
        knn_model = KNeighborsRegressor()
        
        # Define a parameter grid for hyperparameter tuning in KNN Regressor
        knn_param_grid = {
        "n_neighbors": list(range(1, 51, 2)),  # Number of neighbors to consider
        "weights": ["uniform", "distance"],  # Weight function used in predictions
        "p": [1, 2],  # Power parameter for the Minkowski distance
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algorithm used to compute the nearest neighbors
        "leaf_size": list(range(5, 55, 5)),  # Leaf size passed to BallTree or KDTree
        "metric": ["euclidean", "manhattan", "minkowski"],  # Distance metric for the tree
    }
        
        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(knn_model, knn_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a Decision Tree Regressor model with hyperparameter tuning.    
    def tree_regressor_model(self):
        """
        Train a Decision Tree Regressor model with hyperparameter tuning.

        The method informs the user about training the Decision Tree Regressor model and initializes a Decision Tree Regressor model.
        It defines a parameter grid for hyperparameter tuning in Decision Tree Regressor and runs the regressor with hyperparameter
        tuning on the training and testing sets.
        """

        # Inform the user about training the Decision Tree Regressor Model
        print("Training Decision Tree Regressor Model, please wait....")

        # Initialize a Decision Tree Regressor model
        tree_model = DecisionTreeRegressor()

        # Define a parameter grid for hyperparameter tuning in Decision Tree Regressor
        tree_param_grid = {
        "max_depth": [None] + list(range(1, 101, 5)),  # Maximum depth of the tree
        "min_samples_split": [2, 5, 10, 20, 30, 40, 50],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],  # Minimum number of samples required to be at a leaf node
        "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4],  # Minimum weighted fraction of the sum total of weights
        "max_features": ["auto", "sqrt", "log2", None],  # Number of features to consider when looking for the best split
        "ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4],  # Complexity parameter used for Minimal Cost-Complexity Pruning
        "random_state": [101]  # Seed for random number generation
    }

        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(tree_model, tree_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a Random Forest Regressor model with hyperparameter tuning.
    def forest_regressor_model(self):
        """
        Train a Random Forest Regressor model with hyperparameter tuning.

        The method informs the user about training the Random Forest Regressor model and initializes a Random Forest Regressor model.
        It defines a parameter grid for hyperparameter tuning in Random Forest Regressor and runs the regressor with hyperparameter
        tuning on the training and testing sets.
        """

        # Inform the user about training the Random Forest Regressor Model
        print("Training Random Forest Regressor Model, please wait....")

        # Initialize a Random Forest Regressor model
        forest_model = RandomForestRegressor()

        # Define a parameter grid for hyperparameter tuning in Random Forest Regressor
        forest_param_grid = {
                "n_estimators": [50, 100, 200, 300, 400, 500],  # Number of trees in the forest
                "max_depth": [None] + list(range(10, 101, 10)),  # Maximum depth of the trees
                "min_samples_split": [2, 5, 10, 20, 30, 40, 50],  # Minimum number of samples required to split an internal node
                "min_samples_leaf": [1, 2, 4, 8, 16, 32],  # Minimum number of samples required to be at a leaf node
                "bootstrap": [True, False],  # Whether bootstrap samples are used
                "max_features": ["auto", "sqrt", "log2", None],  # Number of features to consider when looking for the best split
                "criterion": ["mse", "mae"],  # Function to measure the quality of a split
                "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3, 0.4],  # Minimum impurity decrease required for a split
                "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4],  # Minimum weighted fraction of the sum total of weights
                "max_samples": [None, 0.5, 0.7, 0.9],  # Number of samples to draw from X to train each base estimator
                "ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4],  # Complexity parameter used for Minimal Cost-Complexity Pruning
                "warm_start": [True, False]  # Whether to reuse the solution of the previous call to fit
        }

        # Run the regressor with hyperparameter tuning using the defined parameter grid
        self.run_regressor(forest_model, forest_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
    
    # Perform hyperparameter tuning, training, and evaluation for a given regression model.
    def run_regressor(self,  model, param_grid, X_train, y_train, X_test, y_test, cv=10):
        """
        Perform hyperparameter tuning, training, and evaluation for a given regression model.

        The method utilizes GridSearchCV to find the best hyperparameters for the model, fits the best model on the
        training data, and evaluates its performance on the test set. Evaluation metrics include Mean Absolute Error (MAE),
        Root Mean Squared Error (RMSE), and R2 Score. The results are stored in the class's model dictionary, and the
        relevant information is printed to the console.

        Args:
        - model (object): The regression model to be trained and evaluated.
        - param_grid (dict): The parameter grid for hyperparameter tuning using GridSearchCV.
        - X_train (DataFrame): The training data features.
        - y_train (Series): The training data target variable.
        - X_test (DataFrame): The test data features.
        - y_test (Series): The test data target variable.
        - cv (int, optional): Number of cross-validation folds for GridSearchCV. Defaults to 10.

        Returns:
        - Tuple: Best hyperparameters, MAE, RMSE, R2 Score, and the best-trained model.
        """
                
        # Record the start time for measuring training duration 
        start_time = time.time()

        # Perform GridSearchCV to find the best hyperparameters for the model
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs= -1)
        grid_search.fit(X_train, y_train)

        # Get the best model and its parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Fit the best model on the training data
        best_model.fit(X_train, y_train)
        # Make predictions on the test set
        predictions = best_model.predict(X_test)

        # Calculate evaluation metrics: RMSE, MAE, R2 Score
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
            
        # Record the end time to calculate elapsed training time
        end_time = time.time()
        elapsed_time = end_time - start_time
            
        # Update the model dictionary with results
        self.model_dict["Regressor"].append(model.__class__.__name__)
        self.model_dict["Best_Parameters"].append(best_params)
        self.model_dict["MAE"].append(mae)
        self.model_dict["RMSE"].append(rmse)
        self.model_dict ["R2_Score"].append(r2)
        self.model_dict["Training_Time"].append(elapsed_time)
        
        # Print the results to the console
        print("*" * 10)
        print(f"Regressor: {model.__class__.__name__}")
        print("Best Parameters:", best_params)
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
        print(f"Training took {elapsed_time:.2f} seconds.")
        print("*"* 10)
        print("")
        
        # Return relevant information for further analysis or use
        return best_params, mae, rmse, r2, best_model
     
    # Compare the performance of different regressors and identify the best-performing regressor.
    def compare_regressor(self):
        """
        Compare the performance of different regressors and identify the best-performing regressor.

        The method iterates through the stored model information in the class's model dictionary and identifies
        the regressor with the lowest Root Mean Squared Error (RMSE). It prints information about the recommended
        regressor, including its name, best parameters, Mean Absolute Error (MAE), RMSE, R2 Score, and training time.
        The information is also stored in the class's attribute 'best_regressor_data.'

        Returns:
        - dict or None: Information about the best-performing regressor, including its name, best parameters,
                       MAE, RMSE, R2 Score, and training time. Returns None if no regressor data is found.
        """

        # Initialize the variable to store the lowest RMSE
        lowest_rmse = float('inf')
        
        # Iterate through the indices of the lists in the model_dict
        for i in range(len(self.model_dict["Regressor"])):
            # Retrieve RMSE value for the current regressor
            rmse = self.model_dict["RMSE"][i]
            
            # Check if the current RMSE is lower than the lowest recorded RMSE
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                # Store information about the best regressor with the lowest RMSE
                best_regressor_data = {
                    "Regressor": self.model_dict["Regressor"][i],
                    "Best_Parameters": self.model_dict["Best_Parameters"][i],
                    "MAE": self.model_dict["MAE"][i],
                    "RMSE": self.model_dict["RMSE"][i],
                    "R2_Score": self.model_dict["R2_Score"][i],
                    "Training_Time": self.model_dict["Training_Time"][i]
                }

        # Check if any regressor data was found
        if best_regressor_data:
            # Print the information about the recommended regressor
            print("Recommended Regressor:")
            print("*" * 10)
            print(f"Regressor: {best_regressor_data['Regressor']}")
            print("Best Parameters:", best_regressor_data['Best_Parameters'])
            print(f"MAE: {best_regressor_data['MAE']}")
            print(f"RMSE: {best_regressor_data['RMSE']}")
            print(f"R2 Score: {best_regressor_data['R2_Score']}")
            print(f"Training took {best_regressor_data['Training_Time']:.2f} seconds.")
            print("*" * 10)

            # Update the attribute with information about the best regressor
            self.best_regressor_data = best_regressor_data

        else:
            # Print a message if no regressor data was found
            print("No regressor data found.")

        # Return information about the best regressor (or None if not found)
        return best_regressor_data
    
    # Saves a version of the best model.
    def save_regressor(self):
        """
        Asks the user if they want to save the recommended regression model and saves it if desired.

        The method checks if there is data for the best regressor and prompts the user to decide whether to save
        the recommended model. If the user chooses to save it, the method extracts the best regressor and its parameters
        from the stored data, creates the model instance, fits it with the provided data, and saves the trained model
        to a specified filename using joblib.dump.

        Returns:
        - dict or None: Information about the saved regressor if it was saved, None otherwise.
        """

        # Infinite loop to keep asking the user until a valid choice is made
        while True:
            # Check if there is no data for the best regressor
            if not self.best_regressor_data:
                print("No best regressor data available.")
                return None

            # Ask the user if they want to save the recommended model
            print("Do you want to save the recommended model?")
            user_input = input("Enter 'yes' or 'no': ").lower()

            # Check the user's input
            if user_input == 'yes':
                # Extract the best regressor and its parameters from the stored data
                best_regressor = self.best_regressor_data['Regressor']
                best_params = self.best_regressor_data['Best_Parameters']

                # Check if the model is a polynomial regression
                if "polynomialfeatures__degree" in best_params:
                    # Create a pipeline for polynomial regression
                    polynomial_model_pipe = Pipeline([
                        ('polynomialfeatures', PolynomialFeatures(degree=best_params["polynomialfeatures__degree"])),
                        ('linearregression', LinearRegression())
                    ])
                    best_model = polynomial_model_pipe
                else:
                    # If not polynomial, instantiate the best regressor class with the best parameters
                    best_model_class = globals()[best_regressor]
                    best_model = best_model_class(**best_params)

                # Fit the best model with the provided data
                best_model.fit(self.X, self.y)

                # Ask the user for the filename to save the model
                filename = input("Enter the filename to save the model (e.g., 'best_regressor_model.pkl'): ")

                # Save the trained model to the specified filename using joblib.dump
                dump(best_model, filename)
                print(f"Trained model saved to {filename}")

                # Return the stored best regressor data
                return self.best_regressor_data

            elif user_input == "no":
                # Inform the user that the recommended model is not saved
                print("Recommended model not saved.")
                return
            else:
                # Inform the user of an invalid choice and continue the loop
                print("Invalid choice. Please enter 'yes' or 'no'.")
                continue


# Classifier class for conducting classification analysis using various classifiers.
class Classifier:
    """
    Classifier class for conducting classification analysis using various classifiers.

    Attributes:
    - data (pandas.DataFrame): The input dataset for classification analysis.
    - target_column (str or list): The target column(s) to predict in the classification.
    - file_path (str): The file path associated with the dataset.
    - X (pandas.DataFrame): The features dataset excluding the target column(s).
    - y (pandas.Series): The target column(s) to predict.
    - X_train, X_test, y_train, y_test (pandas.DataFrame or pandas.Series): Training and testing sets
                                                                             for features and target.
    - model_dict (dict): A dictionary to store information about different classifiers.
                        Keys include "Classifier," "Best_Parameters," "Accuracy," "Precision," "Recall,"
                        "F1_Score," and "Training_Time."
    - best_classifier_data (dict or None): Information about the best-performing classifier, including its name,
                                           best parameters, accuracy, precision, recall, F1 score, and training time.
                                           None if no classifier data is available.

     Methods:
     - __init__(self, data, target_column, file_path): Initialize the Classifier class with provided data, target column, and file path.
    
     - run_all(): Runs multiple classification models and evaluates their performance.
    
     - compare_classifier(): Compares the performance of different classifiers and identifies the best-performing one.
   
     - save_classifier(): Asks the user if they want to save the recommended classification model and saves it if desired.
   
     - logistic_classifier(): Train a Logistic Regression Classifier with hyperparameter tuning and evaluation.
    
     - knn_classifier(): Train a k-Nearest Neighbors (KNN) Classifier and evaluate its performance.
    
     - svc_classifier(): Train a Support Vector Classifier (SVC) and evaluate its performance.
     
     - tree_classifier_model(): Train a Decision Tree Classifier and evaluate its performance.
   
     - forest_classifier_model(): Train a Random Forest Classifier and evaluate its performance.
   
     - run_classifier(model, param_grid, X_train, y_train, X_test, y_test, cv=10): Train a classifier using grid search for hyperparameter tuning and evaluate its performance.
   
     - compare_classifier(): Compare and identify the best-performing classifier based on accuracy.
   
     - save_classifier(): Save the recommended classifier to a file.
    """
    
    # Initialize an instance of Regressor.
    def __init__(self, data, target_column, file_path) -> None:
        """
        Initialize the Regressor class with provided data, target column, and file path.

        Parameters:
        - data (pd.DataFrame): The input data for regression.
        - target_column (str or list): The target column(s) for regression.
        - file_path (str): The file path for saving or loading models.

        Returns:
        None
        """

        # Initialize the class with provided data, target column, and file path
        self.data= data
        self.file_path = file_path

        # If target_column is a list, exclude those columns from X; otherwise, exclude the single target_column
        if isinstance(target_column, list):
            self.X = self.data.drop(columns=target_column)
        else:
            self.X = self.data.drop(columns=[target_column])
        
        # Assign the target column to y
        self.y = self.data[target_column]
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size= 0.3, random_state= 101)

        # Initialize a dictionary to store information about different classifiers
        self.model_dict = { 
            "Classifier": [],
            "Best_Parameters": [] ,
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1_Score": [],
            "Training_Time": [],
            }
        
        # Initialize a variable to store the best classifier data
        self.best_classifier_data = None
    
    # Run multiple regression models and compare their performance.
    def run_all(self):
        """
        Run multiple regression models and compare their performance.

        The method sequentially runs the following regression models:
        - Linear Regression
        - Polynomial Regression
        - Lasso Regression
        - Ridge Regression
        - Elastic Net Regression
        - Support Vector Regressor
        - k-Nearest Neighbors Regressor
        - Decision Tree Regressor
        - Random Forest Regressor

        After running each model, it collects information about their performance metrics and
        compares their results. Finally, the method saves the best-performing regressor based on the comparison.
        """

        # Run logistic regression classifier
        self.logistic_classifier()
        # Run k-nearest neighbors (KNN) classifier
        self.knn_classifier()
        # Run support vector machine (SVM) classifier
        self.svc_classifier()
        # Run decision tree classifier
        self.tree_classifier_model()
        # Run random forest classifier
        self.forest_classifier_model()
        # Compare the performance of all classifiers
        self.compare_classifier()
        # Save the best-performing classifier
        self.save_classifier()

    # Train a Logistic Regression Classifier with hyperparameter tuning and evaluation.
    def logistic_classifier(self):
        """
        Train a Logistic Regression Classifier with hyperparameter tuning and evaluation.

        The method initializes a Logistic Regression model, defines a parameter grid for hyperparameter tuning,
        and then runs the classifier with hyperparameter tuning on the training data. The performance of the
        classifier is evaluated on the testing data, and the results are stored in the class attributes.

        Returns:
        None
        """
        
        # Inform the user about training the Logistic Regression Classifier
        print("Training Logistic Regression Classifier, please wait....\n")

        # Create an instance of Logistic Regression model
        logistics_model = LogisticRegression()

        # Define a parameter grid for hyperparameter tuning
        logistics_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
        "solver": ['lbfgs', 'saga'],  # Algorithm to use for optimization
        "class_weight": [None, 'balanced'],  # Weights associated with classes
        "multi_class": ['auto', 'ovr'],  # Multiclass strategy
        "random_state": [101],  # Seed for random number generation
        "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000],  # Maximum number of iterations
        }
        
        # Run the classifier with hyperparameter tuning and evaluation
        self.run_classifier(logistics_model, logistics_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a k-Nearest Neighbors (KNN) Classifier and evaluate its performance.
    def knn_classifier(self):
        """
        Train a k-Nearest Neighbors (KNN) Classifier and evaluate its performance.

        This method initializes and trains a KNN Classifier model using the provided data. It performs
        hyperparameter tuning with the specified parameter grid and evaluates the model's performance on
        the testing set. The results, including accuracy, precision, recall, F1 score, and training time,
        are recorded in the class's model_dict.

        Parameters:
        None

        Returns:
        None
        """
        
        # Inform the user about training the KNN Classifier
        print("Training KNN Classifier, please wait....")

        # Create an instance of KNN Classifier model
        knn_model = KNeighborsClassifier()

        # Define a parameter grid for hyperparameter tuning
        knn_param_grid = {
        "n_neighbors": list(range(1, 51, 2)),  # Number of neighbors to consider
        "weights": ["uniform", "distance"],  # Weight function used in prediction
        "p": [1, 2],  # Power parameter for Minkowski distance metric
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algorithm to compute nearest neighbors
        "leaf_size": list(range(10, 41, 5)),  # Size of the leaf node in the KD-tree
        "metric": ["euclidean", "manhattan", "minkowski"],  # Distance metric for the tree
        "n_jobs": [-1],  # Number of parallel jobs to run for neighbors search (-1 indicates using all available processors)
        }

        # Run the classifier with hyperparameter tuning and evaluation
        self.run_classifier(knn_model, knn_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
    
    # Train a Support Vector Classifier (SVC) and evaluate its performance.
    def svc_classifier(self):
        """
        Train a Support Vector Classifier (SVC) and evaluate its performance.

        This method initializes and trains a Support Vector Classifier (SVC) model using the provided data.
        It performs hyperparameter tuning with the specified parameter grid and evaluates the model's performance
        on the testing set. The results, including accuracy, precision, recall, F1 score, and training time,
        are recorded in the class's model_dict.

        Parameters:
        None

        Returns:
        None
        """
        
        # Inform the user about training the Support Vector Classifier
        print("Training Support Vector Classifier, please wait....")

        # Create an instance of Support Vector Classifier model
        svc_model = SVC()
        
        # Define a parameter grid for hyperparameter tuning
        svc_param_grid = {
        "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20, 50, 100],  # Regularization parameter
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],  # Specifies the kernel type
        "gamma": ["scale", "auto"],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        "degree": list(range(2, 6)),  # Degree of the polynomial kernel function ('poly' only)
        "coef0": [0.0, 0.1, 0.5, 1.0],  # Independent term in the kernel function
        "shrinking": [True, False],  # Whether to use the shrinking heuristic
        "tol": [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criterion
        "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000],  # Maximum number of iterations
        }   

        # Run the classifier with hyperparameter tuning and evaluation
        self.run_classifier(svc_model, svc_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a Decision Tree Classifier.    
    def tree_classifier_model(self):
        """
        Train a Decision Tree Classifier, perform hyperparameter tuning, and evaluate its performance.

        This method initializes and trains a Decision Tree Classifier using the provided training data (self.X_train and self.y_train).
        It performs hyperparameter tuning with the specified parameter grid and evaluates the model's performance on the testing set (self.X_test and self.y_test).
        The results, including accuracy, precision, recall, F1 score, and training time, are recorded in the class's model_dict.

        Parameters:
        None

        Returns:
        None
        """
        
        # Inform the user about training the Decision Tree Classifier
        print("Training Decision Tree Classifier, please wait....")

        # Create an instance of Decision Tree Classifier model
        tree_model = DecisionTreeClassifier()

        # Define a parameter grid for hyperparameter tuning
        tree_param_grid = {
        "criterion": ["gini", "entropy"],  # Function to measure the quality of a split
        "max_depth": [None] + list(range(5, 101, 5)),  # Maximum depth of the tree
        "min_samples_split": [2, 5, 10, 20, 30],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 2, 4, 8, 12],  # Minimum number of samples required to be at a leaf node
        "max_features": ["auto", "sqrt", "log2", None],  # Number of features to consider for the best split
        "ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4],  # Complexity parameter used for Minimal Cost-Complexity Pruning
        "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3, 0.4],  # A node will be split if this split induces a decrease of the impurity greater than or equal to this value
        "class_weight": [None, "balanced", "balanced_subsample"],  # Weights associated with classes in the form {class_label: weight}
        }

        # Run the classifier with hyperparameter tuning and evaluation
        self.run_classifier(tree_model, tree_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    # Train a Random Forest Classifier.
    def forest_classifier_model(self):
        """
        Train a Random Forest Classifier, perform hyperparameter tuning, and evaluate its performance.

        This method initializes and trains a Random Forest Classifier using the provided training data (self.X_train and self.y_train).
        It performs hyperparameter tuning with the specified parameter grid and evaluates the model's performance on the testing set (self.X_test and self.y_test).
        The results, including accuracy, precision, recall, F1 score, and training time, are recorded in the class's model_dict.

        Parameters:
        None

        Returns:
        None
        """
        
        # Inform the user about training the Random Forest Classifier
        print("Training Random Forest Classifier, please wait....")

        # Create an instance of Random Forest Classifier model
        forest_model = RandomForestClassifier()

        # Define a parameter grid for hyperparameter tuning
        forest_param_grid = {
        "n_estimators": [25, 50, 100, 200, 300],  # Number of trees in the forest
        "criterion": ["gini", "entropy"],  # Function to measure the quality of a split
        "max_depth": [None, 20, 50, 100],  # Maximum depth of the trees in the forest
        "min_samples_split": [2, 5, 10, 20],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 4, 8, 16],  # Minimum number of samples required to be at a leaf node
        "bootstrap": [True, False],  # Whether bootstrap samples are used when building trees
        "max_features": ["auto", "sqrt", "log2", None],  # Number of features to consider for the best split
        "class_weight": [None, "balanced", "balanced_subsample"],  # Weights associated with classes
        "max_samples": [None, 0.8, 0.9, 1.0]  # Proportion of samples used for fitting each individual tree
        }

        # Run the classifier with hyperparameter tuning and evaluation
        self.run_classifier(forest_model, forest_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    #  Train a classifier using grid search for hyperparameter tuning and evaluate its performance.
    def run_classifier(self, model, param_grid, X_train, y_train, X_test, y_test, cv=10):
        """
        Train a classifier using grid search for hyperparameter tuning and evaluate its performance.

        This method performs the following steps:
        1. Flattens y_train and y_test to handle 1D arrays in scikit-learn functions.
        2. Uses grid search for hyperparameter tuning on the provided model.
        3. Fits the best model obtained from grid search with the training data.
        4. Makes predictions on the test data.
        5. Calculates evaluation metrics, including accuracy, precision, recall, and F1 score.
        6. Measures and prints the total training time.
        7. Stores the results in the class's model dictionary.
        8. Displays a visual report, including the best parameters, evaluation metrics, confusion matrix, and a heatmap.

        Parameters:
        - model (object): The classifier model to be trained and evaluated.
        - param_grid (dict): The parameter grid for hyperparameter tuning.
        - X_train (DataFrame): The feature matrix of the training set.
        - y_train (Series): The target labels of the training set.
        - X_test (DataFrame): The feature matrix of the test set.
        - y_test (Series): The target labels of the test set.
        - cv (int, optional): Number of folds for cross-validation in grid search. Default is 10.

        Returns:
        None
        """
         
        # Flatten y_train and y_test to handle 1D arrays in the scikit-learn functions
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        
        # Measure the time taken for training 
        start_time = time.time()

        # Grid Search for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Extract the best model and its parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Fit the best model with the training data
        best_model.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = best_model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        confusion_mat = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)

        # Measure the total time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training took {elapsed_time:.2f} seconds.")

        # Store results in the model dictionary
        self.model_dict["Classifier"].append(model.__class__.__name__)
        self.model_dict["Best_Parameters"].append(best_params)
        self.model_dict["Accuracy"].append(accuracy)
        self.model_dict["Precision"].append(precision)
        self.model_dict["Recall"].append(recall)
        self.model_dict["F1_Score"].append(f1)
        self.model_dict["Training_Time"].append(elapsed_time)
        
        # Display a visual report
        print("*" * 10)
        print(f"Classifier: {model.__class__.__name__}")
        print("Best Parameters:", best_params)

        # Classification Report
        print("\nClassifier Evaluation Metrics:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Confusion Matrix
        print("\nConfusion Matrix:")
        print(confusion_mat)

        # Classification Report
        print("\nClassification Report:")
        print(class_report)

        # Display a heatmap of the Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix{model.__class__.__name__}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        

        print("*" * 10)
        print("")

    # Compare and identify the best-performing classifier based on accuracy.  
    def compare_classifier(self):
        """
        Compare and identify the best-performing classifier based on accuracy.

        This method iterates through the recorded classifier results in the class's model_dict.
        It identifies the classifier with the highest accuracy and provides information about it,
        including the classifier type, best parameters, accuracy, precision, recall, F1 score, and training time.
        The information about the best classifier is printed, and the class attribute 'best_classifier_data'
        is updated with the details of the best-performing classifier.

        Parameters:
        None

        Returns:
        dict or None: A dictionary containing information about the best-performing classifier,
                    including classifier type, best parameters, accuracy, precision, recall, F1 score, and training time.
                    Returns None if no classifier data is found.
        """
        
        # Initialize a variable to track the highest accuracy
        highest_accuracy = 0
        
        # Iterate through the indices of the lists in model_dict
        for i in range(len(self.model_dict["Classifier"])):
            accuracy = self.model_dict["Accuracy"][i]
            
            # Check if the current accuracy is higher than the highest recorded accuracy
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                # Store information about the best classifier found so far
                best_classifier_data = {
                    "Classifier": self.model_dict["Classifier"][i],
                    "Best_Parameters": self.model_dict["Best_Parameters"][i],
                    "Accuracy": self.model_dict["Accuracy"][i],
                    "Precision": self.model_dict["Precision"][i],
                    "Recall": self.model_dict["Recall"][i],
                    "F1_Score": self.model_dict["F1_Score"][i],
                    "Training_Time": self.model_dict["Training_Time"][i]
                }

        # Check if any best classifier data was found
        if best_classifier_data:
            # Print information about the recommended classifier with the highest accuracy
            print("Recommended Classifier:")
            print("*" * 10)
            print(f"Classifier: {best_classifier_data['Classifier']}")
            print("Best Parameters:", best_classifier_data['Best_Parameters'])
            print(f"Accuracy: {best_classifier_data['Accuracy']}")
            print(f"Precision: {best_classifier_data['Precision']}")
            print(f"Recall: {best_classifier_data['Recall']}")
            print(f"F1 Score: {best_classifier_data['F1_Score']}")
            print(f"Training took {best_classifier_data['Training_Time']:.2f} seconds.")
            print("*" * 10)

            # Update the class attribute with the best classifier data
            self.best_classifier_data = best_classifier_data

        else:
            # Inform the user if no classifier data was found
            print("No classifier data found.")

        # Return the information about the best classifier found
        return best_classifier_data

    # Save the recommended classifier to a file.
    def save_classifier(self):
        """
        Save the recommended classifier to a file.

        This method checks if there is any recommended classifier data available and prompts the user
        to save the recommended model. If the user chooses to save the model, it retrieves the recommended
        classifier and its best parameters, creates an instance of the recommended model class with the best
        parameters, trains the model with the available data, and then saves the trained model to a specified
        filename using joblib.dump.

        Parameters:
        None

        Returns:
        dict or None: If the recommended model is saved, returns the best classifier data; otherwise, returns None.
        """
        
        # Keep prompting the user until a valid best classifier data is available.
        while True:
            # Check if there is no best classifier data available.
            if not self.best_classifier_data:
                print("No best classifier data available.")
                return None
            
            # Ask the user if they want to save the recommended model.
            print("Do you want to save the recommended model?")
            user_input = input("Enter 'yes' or 'no': ").lower()

            # Check the user's input.
            if user_input == 'yes':
                # Retrieve the recommended classifier and its best parameters.
                best_classifier = self.best_classifier_data['Classifier']
                best_params = self.best_classifier_data['Best_Parameters']

                # Create an instance of the recommended model class with the best parameters.
                best_model_class = globals()[best_classifier]
                best_model = best_model_class(**best_params)
                # Train the model with the available data.
                best_model.fit(self.X, self.y)

                # Ask the user for a filename to save the trained model.
                filename = input("Enter the filename to save the model (e.g., 'best_model.pkl'): ")
                
                # Save the trained model to the specified filename.
                dump(best_model, filename)
                print(f"Trained model saved to {filename}")

                # Return the best classifier data.
                return self.best_classifier_data
            
            elif user_input == "no":
                # Inform the user that the recommended model is not saved.
                print("Recommended model not saved.")
                return 
            else :
                # Inform the user of an invalid choice and prompt again.
                print("invalid choice please enter a valid choice.")
                continue




