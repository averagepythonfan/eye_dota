from pydantic import BaseModel


class PredictTotal(BaseModel):
    radiant_team_id: int
    dire_team_id: int
    radiant_picks: list[int]
    dire_picks: list[int]
    bank: float
    bet_coef: float
    bias: int
    total_over_coefs: tuple[float, float]
    total_less_coefs: tuple[float, float]
