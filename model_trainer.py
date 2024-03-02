
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import joblib
import shutil
import os



class Regressor:

    def __init__(self, data, target_column) -> None:
        self.data= data
        os.system('cls' if os.name == 'nt' else 'clear')
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
        print("Training LR Model please wait....")
        lr_model = LinearRegression()
        lr_param_grid = {}
        self.run_regressor(lr_model, lr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def polynomial_model(self):
       
        print("Training Poly Model please wait....")
        poly_param_grid = {
            "polynomialfeatures__degree": list(range(1, 10)),
            "polynomialfeatures__include_bias": [False]
        }

        
        polynomial_model_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

        
        self.run_regressor(polynomial_model_pipe, poly_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def lasso_model(self):

        print("Training Lasso Model please wait....")
        lasso_model = Lasso()

        
        alphas = np.logspace(-4, 2, 7)  

        lasso_param_grid = {
            "alpha": alphas
        }
        
        self.run_regressor(lasso_model, lasso_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def ridge_model(self):
        print("Training Ridge Model, please wait....")
        ridge_model = Ridge()

        alphas = np.logspace(-4, 2, 7)
        ridge_param_grid = {
            "alpha": alphas
        }

        self.run_regressor(ridge_model, ridge_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def elastic_model(self):
        print("Training ElasticNet Model, please wait....")
        elastic_model = ElasticNet(max_iter= 10000)

        alphas = np.logspace(-4, 2, 7)
        l1_ratios = np.linspace(0, 1, num=11)  

        elastic_param_grid = {
            "alpha": alphas,
            "l1_ratio": l1_ratios
        }

        self.run_regressor(elastic_model, elastic_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def svr_model(self):
        print("Training SVR Model, please wait....")
        
        svr_model = SVR()
        
        svr_param_grid = {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "C": [0.001,0.01,0.1, 1, 5, 10, 20, 50, 70, 100],
            "epsilon": [0.01, 0.05, 0.1, 1.5, 0.2],
            "gamma": ["auto", "scale"]
    }
    
        self.run_regressor(svr_model, svr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def knn_regressor_model(self):
        print("Training KNN Regressor Model, please wait....")
        
        knn_model = KNeighborsRegressor()
        
        knn_param_grid = {
            "n_neighbors": list(range(1,11)),
            "weights": ["uniform", "distance"],
            "p": [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
        }
        
        self.run_regressor(knn_model, knn_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)
        
    def tree_regressor_model(self):
        pass

    def forest_regressor_model(self):
        pass

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

    def __init__(self) -> None:
        self.data= None

    def logistic_classifier(self):
        pass

    def knn_classifier(self):
        pass

    def svc_classifier(self):
        pass
    
    def tree_classifier_model(self):
        pass

    def forest_classifier_model(self):
        pass

    def run_classifier(model, X_train, y_train, X_test, y_test):
    
        model.fit(X_train,y_train)

        predictions= model.predict(X_test)



