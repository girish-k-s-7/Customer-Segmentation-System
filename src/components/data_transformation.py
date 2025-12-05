import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


class DataTransformation:

    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_preprocessor(self, df):
         

        logging.info("Identifying numerical and categorical columns")

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        logging.info(f"Numerical columns: {numerical_cols}")
        logging.info(f"Categorical columns: {categorical_cols}")

         
        num_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols)
            ]
        )

        logging.info("Preprocessor pipeline created successfully")
        return preprocessor


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("=== Starting Data Transformation ===")

            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info(f"Train shape: {df_train.shape}")
            logging.info(f"Test shape: {df_test.shape}")

             
            preprocessor = self.get_preprocessor(df_train)

             
            logging.info("Fitting preprocessor on training data")
            train_arr = preprocessor.fit_transform(df_train)

            logging.info("Transforming test data")
            test_arr = preprocessor.transform(df_test)

             
            save_object(
                file_path=self.preprocessor_path,
                obj=preprocessor
            )

            logging.info(f"Preprocessor saved at: {self.preprocessor_path}")

            logging.info("Data Transformation Completed Successfully")

            return (
                train_arr,
                test_arr,
                self.preprocessor_path
            )

        except Exception as e:
            logging.error("Error in Data Transformation", exc_info=True)
            raise CustomException(e, sys)
