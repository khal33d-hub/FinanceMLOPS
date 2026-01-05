import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
            }

            params = {
                "Linear Regression": {
                # usually nothing to tune here for basic LinearRegression
                # optionally: "fit_intercept": [True, False]
                },
                "Ridge Regression": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "fit_intercept": [True, False],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga", "lbfgs"],
                "max_iter": [1000, 5000, 10000],
                },
                "Lasso Regression": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
                "max_iter": [1000, 5000, 10000],
                "selection": ["cyclic", "random"],
                }
}
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score>5:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"Best found model is {best_model_name} with r2 score: {best_model_score}")
            logging.info(f"best model parameters: {best_model.get_params()}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            MAE = mean_absolute_error(y_test, predicted)
            return MAE
            



            
        except Exception as e:
            raise CustomException(e,sys)