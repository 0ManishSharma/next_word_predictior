from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout
from dataclasses import dataclass
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
@dataclass



class ModelTrainerConfig:
    model_file_path = "artifacts/model.h5"

class RNNModel:
    def __init__(self):
        self.config = ModelTrainerConfig
        
        os.makedirs(os.path.dirname(self.config.model_file_path),exist_ok=True)


    def build_model(self,vocab_size):
        model = Sequential()

        # Embedding Layers
        model.add(Embedding(
            input_dim = vocab_size,
            output_dim = 128
        ))
        model.add(
            LSTM(128,
                 return_sequences = False
                 )
                
        )
        model.add(Dropout(0.3))

        model.add(Dense(vocab_size,activation="softmax"))

        model.compile(
            optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ['accuracy']
        )

        return model
    def train(self,X_train,y_train,X_test,y_test,vocab_size):
        """
        Trains the RNN/LSTM model
        """
        model = self.build_model(vocab_size)
        early_stopping = EarlyStopping(
            monitor = "val_loss",
            patience =3,
            restore_best_weights = True
        )
        checkpoint = ModelCheckpoint(
            filepath = self.config.model_file_path,
            monitor = "val_loss",
            save_best_only = True
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data = (X_test,y_test),
            epochs =10,
            batch_size = 64,
            callbacks = [early_stopping,checkpoint]

        )
        model.save(self.config.model_file_path)
        

