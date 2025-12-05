import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

class DataIngestion:

    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion for Customer Segmentation")

        try:
             
            source_path = os.path.join(
                "notebooks",
                "data",
                "processed",
                "customer_cleaned.csv"    
            )

            logging.info(f"Reading dataset from: {source_path}")
            df = pd.read_csv(source_path)
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

             
            os.makedirs("artifacts", exist_ok=True)

             
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw dataset saved at: {self.raw_data_path}")

             
            logging.info("Performing train-test split (80/20)")
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)

            logging.info("Train and test files saved successfully.")
            logging.info("=== Data Ingestion Completed ===")

            return (
                self.raw_data_path,
                self.train_data_path,
                self.test_data_path,
            )

        except Exception as e:
            logging.error("Error during data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestor = DataIngestion()
    raw_path, train_path, test_path = ingestor.initiate_data_ingestion()
    print("Raw data path:", raw_path)
    print("Train path:", train_path)
    print("Test path:", test_path)
