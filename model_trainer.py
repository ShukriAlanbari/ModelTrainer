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



class Regressor:

    def __init__(self, data, target_column) -> None:
        self.data= data
       # os.system('cls' if os.name == 'nt' else 'clear')
        if isinstance(target_column, list):
            X = self.data.drop(columns=target_column)
        else:
            X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size= 0.3, random_state= 101)
        
        self.model_dict = { 
                "Regressor": [],
                "Best_Parameters": [] ,
                "MAE": [],
                "RMSE": [],
                "R2_Score": [],
                "Training_Time": [],
                }

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
        self.compare_regressor()

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
            "polynomialfeatures__degree": list(range(1, 5)),
            "polynomialfeatures__include_bias": [True, False],
            "polynomialfeatures__interaction_only": [True, False]
        }
        
        polynomial_model_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

        self.run_regressor(polynomial_model_pipe, poly_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def lasso_model(self):
        print("Training Lasso Model, please wait....")
        
        lasso_model = Lasso()

        alphas = np.logspace(-6, 4, 11)  
        lasso_param_grid = {
            "alpha": alphas,
            "fit_intercept": [True, False],  
            "precompute": [True, False], 
            "positive": [True, False],
            "selection": ['cyclic', 'random'],
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000]
              
        }
        
        self.run_regressor(lasso_model, lasso_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def ridge_model(self):
        print("Training Ridge Model, please wait....")
        
        ridge_model = Ridge()

        alphas = np.logspace(-6, 4, 11)
        ridge_param_grid = {
            "alpha": alphas,
            "fit_intercept": [True, False],  
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000], 
        }

        self.run_regressor(ridge_model, ridge_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def elastic_model(self):
        print("Training ElasticNet Model, please wait....")
        
        elastic_model = ElasticNet()

        alphas = np.logspace(-4, 2, 7)
        l1_ratios = np.linspace(0, 1, num=100)  

        elastic_param_grid = {
            "alpha": alphas,
            "l1_ratio": l1_ratios,
            "fit_intercept": [True, False],
            "selection": ['cyclic', 'random'],
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000], 
            "tol": [1e-4, 1e-3, 1e-2],
            "positive": [True, False], 
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
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000], 
        }
        
        self.run_regressor(svr_model, svr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def knn_regressor_model(self):
        print("Training KNN Regressor Model, please wait....")
        
        knn_model = KNeighborsRegressor()
        
        knn_param_grid = {
        "n_neighbors": list(range(1, 51, 2)),  
        "weights": ["uniform", "distance"],
        "p": [1, 2],  
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": list(range(5, 55, 5)),  
        "metric": ["euclidean", "manhattan", "minkowski"],
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
    
    def run_regressor(self,  model, param_grid, X_train, y_train, X_test, y_test, cv=10):
                
        # Time 
        start_time = time.time()

        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs= -1)
        grid_search.fit(X_train, y_train)


        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_model.fit(X_train, y_train)

        predictions = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
            
        # Time
        end_time = time.time()
        elapsed_time = end_time - start_time
            
        
        self.model_dict["Regressor"].append(model.__class__.__name__)
        self.model_dict["Best_Parameters"].append(best_params)
        self.model_dict["MAE"].append(mae)
        self.model_dict["RMSE"].append(rmse)
        self.model_dict ["R2_Score"].append(r2)
        self.model_dict["Training_Time"].append(elapsed_time)
        

        print("*" * 10)
        print(f"Regressor: {model.__class__.__name__}")
        print("Best Parameters:", best_params)
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
        print(f"Training took {elapsed_time:.2f} seconds.")
        print("*"* 10)
        print("")
        
         
        return best_params, mae, rmse, r2, best_model
    
    def compare_regressor(self):
        lowest_rmse = float('inf')
        best_regressor_data = None
        
        # Iterate through the indices of the lists
        for i in range(len(self.model_dict["Regressor"])):
            rmse = self.model_dict["RMSE"][i]
            
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                best_regressor_data = {
                    "Regressor": self.model_dict["Regressor"][i],
                    "Best_Parameters": self.model_dict["Best_Parameters"][i],
                    "MAE": self.model_dict["MAE"][i],
                    "RMSE": self.model_dict["RMSE"][i],
                    "R2_Score": self.model_dict["R2_Score"][i],
                    "Training_Time": self.model_dict["Training_Time"][i]
                }

        if best_regressor_data:
            print("Recommended Regressor:")
            print("*" * 10)
            print(f"Regressor: {best_regressor_data['Regressor']}")
            print("Best Parameters:", best_regressor_data['Best_Parameters'])
            print(f"MAE: {best_regressor_data['MAE']}")
            print(f"RMSE: {best_regressor_data['RMSE']}")
            print(f"R2 Score: {best_regressor_data['R2_Score']}")
            print(f"Training took {best_regressor_data['Training_Time']:.2f} seconds.")
            print("*" * 10)
        else:
            print("No regressor data found.")


class Classifier:
    
    def __init__(self, data, target_column, file_path) -> None:
        self.data= data
        self.file_path = file_path
      # os.system('cls' if os.name == 'nt' else 'clear')
        
        if isinstance(target_column, list):
            self.X = self.data.drop(columns=target_column)
        else:
            self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size= 0.3, random_state= 101)

         
        self.model_dict = { 
            "Classifier": [],
            "Best_Parameters": [] ,
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1_Score": [],
            "Training_Time": [],
            }
        
        self.best_classifier_data = None

    def run_all(self):
        self.logistic_classifier()
        self.knn_classifier()
        # self.svc_classifier()
        # self.tree_classifier_model()
        # self.forest_classifier_model()
        self.compare_classifier()
        self.save_classifier()

    def logistic_classifier(self):
        print("Training Logistic Regression Classifier, please wait....\n")

        logistics_model = LogisticRegression()

        logistics_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        # "solver": ['lbfgs', 'saga'],
        # "class_weight": [None, 'balanced'],
        # "multi_class": ['auto', 'ovr'],
        # "random_state": [101],
        # "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000], 
}

        
        self.run_classifier(logistics_model, logistics_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def knn_classifier(self):
        print("Training KNN Classifier, please wait....")

        knn_model = KNeighborsClassifier()

        knn_param_grid = {
            "n_neighbors": list(range(1, 51, 2)),
            # "weights": ["uniform", "distance"],
            # "p": [1, 2],
            # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            # "leaf_size": list(range(10, 41, 5)),
            # "metric": ["euclidean", "manhattan", "minkowski"],
            # "n_jobs": [-1],
            }

        self.run_classifier(knn_model, knn_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
   
    def svc_classifier(self):
        print("Training Support Vector Classifier, please wait....")

        svc_model = SVC()

        svc_param_grid = {
            "C": [0.001, 0.01, 0.1,0.5, 1, 5, 10, 20, 50, 100],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ["scale", "auto"],
            "degree": list(range(2, 6)),
            "coef0": [0.0, 0.1, 0.5, 1.0],
            "shrinking": [True, False],
            "tol": [1e-4, 1e-3, 1e-2],
            "max_iter": [100, 1000, 10000, 100000, 1000000, 10000000],
            }

        self.run_classifier(svc_model, svc_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
        
    def tree_classifier_model(self):
        print("Training Decision Tree Classifier, please wait....")

        tree_model = DecisionTreeClassifier()

        tree_param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None] + list(range(5, 101, 5)),  
            "min_samples_split": [2, 5, 10, 20, 30],  
            "min_samples_leaf": [1, 2, 4, 8, 12],  
            "max_features": ["auto", "sqrt", "log2", None],
            "ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4],
            "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3, 0.4],  
            "class_weight": [None, "balanced", "balanced_subsample"]
            }
        self.run_classifier(tree_model, tree_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def forest_classifier_model(self):
        print("Training Random Forest Classifier, please wait....")

        forest_model = RandomForestClassifier()

        forest_param_grid = {
            "n_estimators": [25, 50, 100, 200, 300],  
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 20, 50, 100],  
            "min_samples_split": [2, 5, 10, 20],  
            "min_samples_leaf": [1, 4, 8, 16],  
            "bootstrap": [True, False],
            "max_features": ["auto", "sqrt", "log2", None],
            "class_weight": [None, "balanced", "balanced_subsample"],
            "max_samples": [None, 0.8, 0.9, 1.0]
            }

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
        # Time taken for training
        elapsed_time = end_time - start_time
        print(f"Training took {elapsed_time:.2f} seconds.")


        self.model_dict["Classifier"].append(model.__class__.__name__)
        self.model_dict["Best_Parameters"].append(best_params)
        self.model_dict["Accuracy"].append(accuracy)
        self.model_dict["Precision"].append(precision)
        self.model_dict["Recall"].append(recall)
        self.model_dict["F1_Score"].append(f1)
        self.model_dict["Training_Time"].append(elapsed_time)
        
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

        

        print("*" * 10)
        print("")
        
    def compare_classifier(self):
        highest_accuracy = 0
        
        # Iterate through the indices of the lists
        for i in range(len(self.model_dict["Classifier"])):
            accuracy = self.model_dict["Accuracy"][i]
            
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_classifier_data = {
                    "Classifier": self.model_dict["Classifier"][i],
                    "Best_Parameters": self.model_dict["Best_Parameters"][i],
                    "Accuracy": self.model_dict["Accuracy"][i],
                    "Precision": self.model_dict["Precision"][i],
                    "Recall": self.model_dict["Recall"][i],
                    "F1_Score": self.model_dict["F1_Score"][i],
                    "Training_Time": self.model_dict["Training_Time"][i]
                }

        if best_classifier_data:
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

            self.best_classifier_data = best_classifier_data

        else:
            print("No classifier data found.")
        return best_classifier_data

    def save_classifier(self):
        while True:
            if not self.best_classifier_data:
                print("No best classifier data available.")
                return None
            
            print("Do you want to save the recommended model?")
            user_input = input("Enter 'yes' or 'no': ").lower()
        
            if user_input == 'yes':
                best_classifier = self.best_classifier_data['Classifier']
                best_params = self.best_classifier_data['Best_Parameters']

                best_model_class = globals()[best_classifier]
                best_model = best_model_class(**best_params)
                best_model.fit(self.X, self.y)

                filename = input("Enter the filename to save the model (e.g., 'best_model.pkl'): ")
                
                # dump_path = os.path.join(self.file_path, filename)

                dump(best_model, filename)
                print(f"Trained model saved to {filename}")

                return self.best_classifier_data
            
            elif user_input == "no":
                print("Recommended model not saved.")
                return 
            else :
                print("invalid choice please enter a valid choice.")
                continue




