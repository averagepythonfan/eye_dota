from services import PredictTotalModel, MongoService
from config import SCALER_PATH, MODEL_PATH, MONGO


def get_predict_total_model():
    return PredictTotalModel(model_path=MODEL_PATH)

def get_mongo_service():
    return MongoService(uri=MONGO)
