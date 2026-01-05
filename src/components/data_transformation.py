import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Build and return a preprocessing object 
        """
        try:
            feature_columns = ["Open", "Low", "High", "Volume", "AdjClose"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, feature_columns),
                ],
                remainder="drop",
            )

            logging.info(f"Numerical feature columns for preprocessing: {feature_columns}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column_name = "Close"
            feature_columns = ["Open", "Low", "High", "Volume", "AdjClose"]

            # if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
            #     raise ValueError(f"Target column '{target_column_name}' not found in train/test data")

            # missing_train_features = [c for c in feature_columns if c not in train_df.columns]
            # missing_test_features = [c for c in feature_columns if c not in test_df.columns]
            # if missing_train_features or missing_test_features:
            #     raise ValueError(
            #         f"Missing feature columns. "
            #         f"Train missing={missing_train_features}, Test missing={missing_test_features}"
            #     )

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
