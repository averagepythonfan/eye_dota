from services import PredictTotalModel, MongoService, AiohttpService, RidgeDurationModel
from config import SCALER_PATH, MODEL_PATH, MONGO, RIDGE_MODEL, RIDGE_SCALER


def get_predict_total_model():
    return PredictTotalModel(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

def get_mongo_service():
    return MongoService(uri=MONGO)


def get_aiohttp_service():
    return AiohttpService()


def get_rigde_duration():
    return RidgeDurationModel(model_path=RIDGE_MODEL, scaler_path=RIDGE_SCALER)
