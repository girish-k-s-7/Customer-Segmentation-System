import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


class ModelTrainer:

    def __init__(self, artifact_dir: str = "artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.model_path = os.path.join(self.artifact_dir, "kmeans.pkl")

    def _select_k_by_silhouette(self, X, k_min=2, k_max=8):
        logger.info("Selecting best k using silhouette score")
        best_k = None
        best_score = -1
        scores = {}
        for k in range(k_min, min(k_max, X.shape[0] - 1) + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                if len(np.unique(labels)) == 1:
                    score = -1
                else:
                    score = silhouette_score(X, labels)
                scores[k] = score
                logger.info(f"k={k} silhouette={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.warning(f"Failed to evaluate k={k}: {e}")
        logger.info(f"Selected k={best_k} with silhouette={best_score:.4f}")
        return best_k, best_score, scores

    def initiate_model_trainer(self, train_array, test_array=None, best_k: int = None):
        
        try:
            logger.info("Starting model training")

            X = train_array
            if hasattr(X, "toarray"):  
                X = X.toarray()

             
            if best_k is None:
                selected_k, best_score, scores = self._select_k_by_silhouette(X, k_min=2, k_max=8)
                if selected_k is None:
                     
                    selected_k = 4
                    logger.warning("Could not select k via silhouette; falling back to k=4")
            else:
                selected_k = best_k
                best_score = None
                scores = {}

            logger.info(f"Training final KMeans with k={selected_k}")
            kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(X)

             
            test_score = None
            if test_array is not None:
                Xt = test_array
                if hasattr(Xt, "toarray"):
                    Xt = Xt.toarray()
                try:
                    pred_test = kmeans.predict(Xt)
                    if len(np.unique(pred_test)) > 1:
                        test_score = silhouette_score(Xt, pred_test)
                        logger.info(f"Silhouette on test set: {test_score:.4f}")
                except Exception as e:
                    logger.warning(f"Could not compute silhouette on test set: {e}")

            # Save model
            save_object(self.model_path, kmeans)
            logger.info(f"KMeans model saved at: {self.model_path}")

            info = {
                "k": selected_k,
                "train_silhouette": best_score,
                "test_silhouette": test_score,
                "silhouette_scores": scores
            }

            logger.info("=== Model training completed ===")
            return self.model_path, info

        except Exception as e:
            logger.error("Error in model trainer", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=4, random_state=42)
    trainer = ModelTrainer()
    model_path, info = trainer.initiate_model_trainer(X)
    print("Model path:", model_path)
    print("Info:", info)
