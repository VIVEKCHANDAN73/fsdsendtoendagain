import os 
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.DimondPricePrediction.utils.utils import load_object

class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse,mae,r2
    
    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)

            mlflow.set_registry_uri("https://dagshub.com/VIVEKCHANDAN73/fsdsendtoendagain.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = model.predict(X_test)
                (rmse,mae,r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("mae",mae)
                mlflow.log_metric("r2",r2)

                # this condition is for dagshub
                # model registry does not work with file store
                if tracking_url_type_store != 'file':
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case
                    mlflow.sklearn.log_model(model,"model",registered_model_name="ml_model")
                # this condition is for local
                else:
                    mlflow.sklearn.log_model(model,"model")

        except Exception as e:
            raise e
