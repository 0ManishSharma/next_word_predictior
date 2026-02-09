import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

class DataTransformation:
    def __init__(self):
        self.tokenizer= Tokenizer()
        self.max_length = 100
        self.tokenizer_path = "artifacts/tokenizer.pkl"

    def clean_text(self,text):
        """
        Clean input text
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    def transfromed_data(self,file_path):
        """
        Reads file and transforms text into padded sequences
        """

        with open(file_path,'r',encoding='utf-8') as f:
            texts = f.readlines()

        cleaned_text = [self.clean_text(text) for text in texts]

        self.tokenizer.fit_on_texts(cleaned_text)

        sequence = self.tokenizer.texts_to_sequences(cleaned_text)

        # Padding

        padded_sequences = pad_sequences(
            sequence,
            maxlen = self.max_length,
            padding = "post"

        )
        X = padded_sequences[:,:-1]
        y = padded_sequences[:,-1]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        with open(self.tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        return X_train,X_test,y_train,y_test

