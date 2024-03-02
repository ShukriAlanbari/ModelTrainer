
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
        print("Training LR Model please wait....\n")
        lr_model = LinearRegression()
        lr_param_grid = {}
        self.run_regressor(lr_model, lr_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def polynomial_model(self):
       
        print("Training Poly Model please wait....\n")
        poly_param_grid = {
            "polynomialfeatures__degree": list(range(1, 10)),
            "polynomialfeatures__include_bias": [False]
        }

        
        polynomial_model_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

        
        self.run_regressor(polynomial_model_pipe, poly_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def lasso_model(self):

        print("Training Lasso Model please wait....\n")
        lasso_model = Lasso()

        
        alphas = np.logspace(-4, 1, 6)  

        lasso_param_grid = {
            "alpha": alphas
        }
        
        self.run_regressor(lasso_model, lasso_param_grid, self.X_train, self.y_train, self.X_test, self.y_test)

    def ridge_model(self):
        print("Training ridge Model please wait....")
        pass

    def elastic_model(self):
        pass

    def svr_model(self):
        pass

    def knn_regressor_model(self):
        pass
    
    def tree_regressor_model(self):
        pass

    def forest_regressor_model(self):
        pass

    def run_regressor(self, model, param_grid, X_train, y_train, X_test, y_test, cv=10):
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_model.fit(X_train, y_train)

        predictions = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("")
        print("*" * 10)
        print("Best Parameters:", best_params)
        print("")
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