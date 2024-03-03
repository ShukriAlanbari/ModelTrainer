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
import joblib
import shutil
import os
import time



class Regressor:

    def __init__(self, data, target_column) -> None:
        self.data= data
    #   os.system('cls' if os.name == 'nt' else 'clear')
        if isinstance(target_column, list):
            X = self.data.drop(columns=target_column)
        else:
            X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size= 0.3, random_state= 101)
        
    def run_all(self):
        self.linear_model()
        self.polynomial_model()
        self.lasso_model()
        self.ridge_model()
        self.elastic_model()
        self.svr_model()
        self. knn_regressor_model()
        self.tree_regressor_model()
        self.forest_regressor_model()

    def linear_model(self):
        print("Training Linear Regression Model, please wait....")
        
        lr_model = LinearRegression()
        
        lr_param_grid = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],  
        }

        self.run_regressor(lr_model, lr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def polynomial_model(self):
        print("Training Polynomial Regression Model, please wait....")
        
        poly_param_grid = {
            "polynomialfeatures__degree": list(range(1, 10)),
            "polynomialfeatures__include_bias": [True, False],
            "polynomialfeatures__interaction_only": [True, False]
        }
        
        polynomial_model_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

        self.run_regressor(polynomial_model_pipe, poly_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def lasso_model(self):
        print("Training Lasso Model, please wait....")
        
        lasso_model = Lasso()

        alphas = np.logspace(-4, 2, 7)  
        lasso_param_grid = {
            "alpha": alphas,
            "fit_intercept": [True, False],  # Whether to calculate the intercept for this model
            "normalize": [True, False],  # Whether to normalize the features before fitting the model
            "precompute": [True, False],  # Whether to use precomputed Gram matrix for faster calculations
            "positive": [True, False]  # Whether to constrain the coefficients to be non-negative
        }
        
        self.run_regressor(lasso_model, lasso_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def ridge_model(self):
        print("Training Ridge Model, please wait....")
        
        ridge_model = Ridge()

        alphas = np.logspace(-4, 2, 7)
        ridge_param_grid = {
            "alpha": alphas,
            "fit_intercept": [True, False],  
            "normalize": [True, False], 
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']  
        }

        self.run_regressor(ridge_model, ridge_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def elastic_model(self):
        print("Training ElasticNet Model, please wait....")
        
        elastic_model = ElasticNet(max_iter=1000)

        alphas = np.logspace(-4, 2, 7)
        l1_ratios = np.linspace(0, 1, num=11)  

        elastic_param_grid = {
            "alpha": alphas,
            "l1_ratio": l1_ratios,
            "fit_intercept": [True, False],  
            "normalize": [True, False],  
            "selection": ['cyclic', 'random']  
        }

        self.run_regressor(elastic_model, elastic_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
    
    def svr_model(self):
        print("Training SVR Model, please wait....")
        
        svr_model = SVR()
        
        svr_param_grid = {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "C": [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100],
            "epsilon": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0],
            "gamma": ["auto", "scale"],
            "degree": list(range(2, 6)),  
            "coef0": [0.0, 0.1, 0.5, 1.0],  
            "shrinking": [True, False],  
            "tol": [1e-4, 1e-3, 1e-2],  
            "max_iter": [100, 500, 1000]  
        }
        
        self.run_regressor(svr_model, svr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def knn_regressor_model(self):
        print("Training KNN Regressor Model, please wait....")
        
        knn_model = KNeighborsRegressor()
        
        knn_param_grid = {
        "n_neighbors": list(range(1, 31, 2)),  
        "weights": ["uniform", "distance"],
        "p": [1, 2],  
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": list(range(10, 41, 5)),  
        "metric": ["euclidean", "manhattan", "minkowski"]
    }
        
        self.run_regressor(knn_model, knn_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
        
    def tree_regressor_model(self):
        print("Training Decision Tree Regressor Model, please wait....")

        tree_model = DecisionTreeRegressor()

        tree_param_grid = {
        "max_depth": [None] + list(range(1, 101, 5)),
        "min_samples_split": [2, 5, 10, 20, 30, 40, 50],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4],
        "max_features": ["auto", "sqrt", "log2", None],
        "ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4],
        "random_state": [101]  
    }

        self.run_regressor(tree_model, tree_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def forest_regressor_model(self):
        print("Training Random Forest Regressor Model, please wait....")

        forest_model = RandomForestRegressor()

        forest_param_grid = {
                            "n_estimators": [50, 100, 200, 300, 400, 500],
                            "max_depth": [None] + list(range(10, 101, 10)),
                            "min_samples_split": [2, 5, 10, 20, 30, 40, 50],
                            "min_samples_leaf": [1, 2, 4, 8, 16, 32],
                            "bootstrap": [True, False],
                            "max_features": ["auto", "sqrt", "log2", None],
                            "criterion": ["mse", "mae"],
                            "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3, 0.4],
                            "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4],
                            "max_samples": [None, 0.5, 0.7, 0.9],
                            "ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4],
                            "warm_start": [True, False]
                        }

        self.run_regressor(forest_model, forest_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
    
    def run_regressor(self, model, param_grid, X_train, y_train, X_test, y_test, cv=10):
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs= -1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_model.fit(X_train, y_train)

        predictions = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        
        print("*" * 10)
        print("Best Parameters:", best_params)
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
        print("*"* 10)
        print("")
        


class Classifier:

    def __init__(self, data, target_column) -> None:
        self.data= data
      # os.system('cls' if os.name == 'nt' else 'clear')
        if isinstance(target_column, list):
            X = self.data.drop(columns=target_column)
        else:
            X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size= 0.3, random_state= 101)

    def run_all(self):
        #self.logistic_classifier()
        #self.knn_classifier()
        #self.svc_classifier()
        #self.tree_classifier_model()
         self.forest_classifier_model()

    def logistic_classifier(self):
        print("Training Logistic Regression Classifier, please wait....\n")

        logistics_model = LogisticRegression()

        logistics_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "max_iter": [100, 500, 1000, 10000],
        "solver": ['lbfgs', 'saga'],
        "class_weight": [None, 'balanced'],
        "multi_class": ['auto', 'ovr'],
        "random_state": [101],}

        
        self.run_classifier(logistics_model, logistics_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def knn_classifier(self):
        print("Training KNN Classifier, please wait....")

        knn_model = KNeighborsClassifier()

        knn_param_grid = {
            "n_neighbors": list(range(1, 31, 2)),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(range(10, 41, 5)),
            "metric": ["euclidean", "manhattan", "minkowski"],
            "n_jobs": [-1],}

        self.run_classifier(knn_model, knn_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
   
    def svc_classifier(self):
        print("Training Support Vector Classifier, please wait....")

        svc_model = SVC()

        svc_param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 5, 10, 20],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ["scale", "auto"],
            "degree": list(range(2, 6)),
            "coef0": [0.0, 0.1, 0.5, 1.0],
            "shrinking": [True, False],
            "tol": [1e-4, 1e-3, 1e-2],
            "max_iter": [100, 500, 1000, 10000],}

        self.run_classifier(svc_model, svc_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
        
    def tree_classifier_model(self):
        print("Training Decision Tree Classifier, please wait....")

        tree_model = DecisionTreeClassifier()

        tree_param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None] + list(range(5, 51, 5)),
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["auto", "sqrt", "log2", None],
            "ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4]}
        self.run_classifier(tree_model, tree_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def forest_classifier_model(self):
        print("Training Random Forest Classifier, please wait....")

        forest_model = RandomForestClassifier()

        forest_param_grid = {
            "n_estimators": [25, 50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 20, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 4, 8],
            "bootstrap": [True, False],
            "max_features": ["auto", "sqrt", "log2", None],
            "class_weight": [None],}

        self.run_classifier(forest_model, forest_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def run_classifier(self, model, param_grid, X_train, y_train, X_test, y_test, cv=10):
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        
        # Time 
        start_time = time.time()

        # Grid Search
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Best Model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Fit Best Model
        best_model.fit(X_train, y_train)

        # Predictions
        predictions = best_model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        confusion_mat = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)

        # Time
        end_time = time.time()

        # Visual Report
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

        # Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix{model.__class__.__name__}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Time taken for training
        elapsed_time = end_time - start_time
        print(f"Training took {elapsed_time:.2f} seconds.")

        print("*" * 10)
        print("")


