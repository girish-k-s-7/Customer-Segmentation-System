import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.logger import logger
from src.exception import CustomException


def run_training_pipeline():
    try:
        logger.info("Training pipeline started")

        # Data ingestion
        ingestion = DataIngestion()
        raw_path, train_path, test_path = ingestion.initiate_data_ingestion()
        logger.info(f"Data ingested: raw={raw_path}, train={train_path}, test={test_path}")

        # Data transformation
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        logger.info(f"Preprocessor saved at: {preprocessor_path}")

        # Model training
        trainer = ModelTrainer()
        model_path, info = trainer.initiate_model_trainer(train_arr, test_arr, best_k=None)

        logger.info("Training pipeline completed successfully")
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Training info: {info}")

        print("Training pipeline finished.")
        print("Model path:", model_path)
        print("Info:", info)

        return {
            "raw_path": raw_path,
            "train_path": train_path,
            "test_path": test_path,
            "preprocessor_path": preprocessor_path,
            "model_path": model_path,
            "info": info
        }

    except Exception as e:
        logger.error("Error in training pipeline", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
