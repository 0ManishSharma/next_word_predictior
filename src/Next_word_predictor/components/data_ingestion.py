from dataclasses import dataclass
import requests
import os


@dataclass
class DataIngestionConfig:
    dataset_url: str = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    raw_text_path : str = os.path.join("artifacts",'raw.txt')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.config.raw_text_path),exist_ok=True)

    def download_data(self):
        print("Download dataset")

        response = requests.get(self.config.dataset_url)
        response.raise_for_status()

        with open (self.config.raw_text_path,'w',encoding="utf-8") as f:
            f.write(response.text)

        print("✅ Dataset downloaded and saved at:", self.config.raw_text_path)

        