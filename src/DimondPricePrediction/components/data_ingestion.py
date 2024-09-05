import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import os
import sys

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")

        try:
            data = pd.read_csv(os.path.join("notebooks/data","gemstone.csv"))
            logging.info("I have read dataset as df")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('I have saved the raw dataset in artifact folder')

            

            logging.info("here I've performed train test split")
            train_data,test_data = train_test_split(data,test_size=0.25)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            logging.info('I have saved the train dataset in artifacts folder')

            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info('I have saved the test dataset in artifacts folder')
            
            logging.info("train test split completed")

        except Exception as e:
            logging.info('Exception occured during data ingestion stage')
            raise customexception(e,sys)