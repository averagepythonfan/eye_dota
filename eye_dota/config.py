import os


MODEL_PATH: str = os.getenv("MODEL_PATH") 
SCALER_PATH: str = os.getenv("SCALER_PATH")
MONGO: str = os.getenv("MONGO")

RIDGE_MODEL: str = os.getenv("RIDGE_MODEL")
RIDGE_SCALER: str = os.getenv("RIDGE_SCALER")

CURRENT_PATCH: str = str(os.getenv("CURRENT_PATCH", "7.36"))
TOTAL_STATS_COEF: float = float(os.getenv("TOTAL_STATS_COEF", 0.7))
DURATIONS_STATS_COEF: float = float(os.getenv("DURATIONS_STATS_COEF", 0.6))
