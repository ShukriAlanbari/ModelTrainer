from data_processing import data, target_column
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import shutil
import os



class Regressor:
    def __init__(self) -> None:
        self.data= None
        
        X_train, X_test, y_train, y_test = train_test_split()

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
        pass

    def polynomial_model(self):
        pass

    def lasso_model(self):
        pass
    
    def ridge_model(self):
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

    def run_regressor(self,model, X_train, y_train, X_test, y_test):

    model.fit(X_train,y_train)

    predictions= model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test,predictions))
    mae = mean_absolute_error(y_test,predictions)

    print(f"RMSE : {rmse}")
    print(f"MAE : {mae}")

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