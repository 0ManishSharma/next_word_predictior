import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Text_predictor:
    def __init__(self):
        
        self.model = load_model("artifacts/model.h5")
        with open ("artifacts/tokenizer.pkl",'rb') as file:
            self.tokenizer = pickle.load(file)

    def clean_text(self, text):
        """
        Cleans input text same as training
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def predict_next_word(self,text):

        cleaned_text = self.clean_text(text)

        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        padded_sequence = pad_sequences(
            sequence,
            maxlen =100,
            padding ='post'
        )
        prediction = self.model.predict(padded_sequence,verbose = 0)
        predicted_index = np.argmax(prediction,axis=-1)[0]
        for word, index in self.tokenizer.word_index.items():
            if index == predicted_index:
                return word
        return None
        