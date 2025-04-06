from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessor
from logger import logging
import os



root_dir="artifacts"
url="https://drive.google.com/file/d/1g6MLOGfOqSum7XhvMo8Dggk2ydKhTg7s/view?usp=sharing"
prefix="https://drive.google.com/uc?id="
data_ingestion_root_dir="artifacts/data_ingestion"
dataset_dir="artifacts/data_ingestion/Dataset.csv"


logging.info(f"Creating directory at {root_dir}")
os.makedirs(root_dir,exist_ok=True)

STAGE_NAME="DATA INGESTION STAGE"
try:
    logging.info(f"{STAGE_NAME} is started")
    obj=DataIngestion(
        url=url,
        prefix=prefix,
        dataset_dir=dataset_dir,
        root_dir=data_ingestion_root_dir
    )
    obj.main()
    logging.info(f"{STAGE_NAME} is Completed")
except Exception as e:
    logging.info(f"{e}")


STAGE_NAME="DATA PREPROCESSING STAGE"
try:
    logging.info(f"{STAGE_NAME} is started")
    obj=DataPreprocessor(
         root_dir="artifacts/data_preprocessing",
         preprocess_data_path="artifacts/data_preprocessing/preprocess_data.csv",
         dataset_dir=dataset_dir
    )
    obj.main()
    logging.info(f"{STAGE_NAME} is Completed")
except Exception as e:
    logging.info(f"{e}")