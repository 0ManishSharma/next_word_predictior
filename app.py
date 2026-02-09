from src.Next_word_predictor.components.data_ingestion import DataIngestion
from src.Next_word_predictor.components.data_transformation import DataTransformation
from src.Next_word_predictor.components.model_tranier import RNNModel
from src.Next_word_predictor.pipelines.prediction_pipeline import Text_predictor


if __name__ == "__main__":
    # data_ingestion = DataIngestion()
    # data_ingestion.download_data()

    # data_trans = DataTransformation()
    # X_train,X_test,y_train,y_test=data_trans.transfromed_data(file_path="artifacts/raw.txt")
    # print("Data Transformation Successfully")
    # model = RNNModel()
    # model.train(X_train,y_train,X_test,y_test,vocab_size=10000)

    predictor = Text_predictor()
    predictor_word = predictor.predict_next_word("wholesome, we might guess they relieved us")
    print("the next word is:",predictor_word)




