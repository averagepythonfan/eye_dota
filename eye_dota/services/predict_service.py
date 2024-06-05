import joblib
import pathlib
import numpy as np
from keras.models import load_model, Sequential


class Singleton:
    _instance = None

    def __new__(cls, model_path, scaler_path):
        if not isinstance(cls._instance, cls):
            cls._instance = super(__class__, cls).__new__(cls)
        return cls._instance    


class PredictTotalModel(Singleton):

    model = None
    index_to_delete = [0, 24, 115, 116, 117, 118, 122, 124,
        125, 127, 130, 131, 132, 133, 134, 139, 140, 141, 142, 143,
        144, 145, 146, 147, 148, 149, 150, 174, 265, 266, 267, 268,
        272, 274, 275, 277, 280, 281, 282, 283, 284, 289, 290, 291,
        292, 293, 294, 295, 296, 297, 298, 299
    ]


    def __init__(self, model_path, scaler_path):
        self.model: Sequential = load_model(model_path)
        self.t_scaler = joblib.load(scaler_path)
    

    def is_load(self):
        return True if self.model else False


    def predict(self, radiant_heroes: list[int], dire_heroes: list[int]):
        assert self.model is not None

        rp, dp = np.array([0 for _ in range(150)]), np.array([0 for _ in range(150)])
        rp[radiant_heroes] = 1
        dp[dire_heroes] = 1
        pic_vec = np.concatenate([rp, dp])
        pic_vec = np.delete(pic_vec, self.index_to_delete)

        preds = self.model.predict(pic_vec.reshape(1, 248), verbose=0)
        return self.t_scaler.inverse_transform(preds).astype(np.int32)[0][0]


class RidgeDurationModel(Singleton):

    ridge = None

    index_to_delete = [0, 24, 115, 116, 117, 118, 122, 124,
        125, 127, 130, 131, 132, 133, 134, 139, 140, 141, 142, 143,
        144, 145, 146, 147, 148, 149, 150, 174, 265, 266, 267, 268,
        272, 274, 275, 277, 280, 281, 282, 283, 284, 289, 290, 291,
        292, 293, 294, 295, 296, 297, 298, 299
    ]


    def __init__(self, model_path, scaler_path) -> None:
        self.ridge = joblib.load(model_path)
        self.ridge_scaler = joblib.load(scaler_path)


    def predict(self, radiant_heroes: list[int], dire_heroes: list[int]):
        assert self.ridge is not None

        rp, dp = np.array([0 for _ in range(150)]), np.array([0 for _ in range(150)])
        rp[radiant_heroes] = 1
        dp[dire_heroes] = 1
        pic_vec = np.concatenate([rp, dp])
        pic_vec = np.delete(pic_vec, self.index_to_delete)

        preds = self.ridge.predict(pic_vec.reshape(1, 248))
        return self.ridge_scaler.inverse_transform(preds).astype(np.int32).flatten()[0]
