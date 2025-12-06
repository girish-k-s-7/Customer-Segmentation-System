import sys
import pandas as pd
import numpy as np
from src.utils import load_object
from src.logger import logger
from src.exception import CustomException


class SegmentPredictor:
   

    def __init__(self, model_path: str = "artifacts/kmeans.pkl", preprocessor_path: str = "artifacts/preprocessor.pkl"):
        try:
            logger.info("Loading model and preprocessor for prediction")
            self.kmeans = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            logger.info("Loaded model and preprocessor successfully")
        except Exception as e:
            logger.error("Error loading model/preprocessor", exc_info=True)
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame):
        
        try:
            logger.info(f"Running prediction for input shape: {input_df.shape}")

            X_trans = self.preprocessor.transform(input_df)
           
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()

            labels = self.kmeans.predict(X_trans)
            distances = self.kmeans.transform(X_trans)  
            min_dist = distances.min(axis=1)

            logger.info("Prediction completed")
            return labels, distances, min_dist

        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    
    try:
        test_path = "artifacts/train.csv"
        df_test = pd.read_csv(test_path)
        sample = df_test.head(5).copy()

        predictor = SegmentPredictor()
        labels, distances, min_dist = predictor.predict(sample)

        print("Labels:", labels)
        print("Min distances:", np.round(min_dist, 4))

    except Exception as e:
        raise CustomException(e, sys)
