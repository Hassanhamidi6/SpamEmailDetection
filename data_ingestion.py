from pathlib import Path
import pandas as pd
import gdown
import os 

from logger import logging

class DataIngestion:
    def __init__(self,
                 root_dir:Path,
                 dataset_dir:Path,
                 url:str,
                 prefix:str):
        
        self.root_dir=root_dir
        self.url=url
        self.dataset_dir=dataset_dir
        self.prefix=prefix

    def main(self):
        try:
            id=self.url.split('/')[5]
            logging.info(f"Creating Directory at {self.root_dir}")
            os.makedirs(self.root_dir,exist_ok=True)
            logging.info(f"Downloading Data From {self.url} into {self.dataset_dir}")
            gdown.download(self.prefix+id,self.dataset_dir)
        except Exception as e:
            logging.error(e)

