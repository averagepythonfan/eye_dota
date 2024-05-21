import numpy as np
from typing import Annotated
from fastapi import APIRouter, Depends
from schemas import PredictTotal
from services import PredictTotalModel, MongoService
from dependencies import get_predict_total_model, get_mongo_service


main = APIRouter(
    prefix="/main",
    tags=["Main"]
)


@main.post("/predict_total")
async def predict_total(
    predict_data: PredictTotal,
    model_service: Annotated[PredictTotalModel, Depends(get_predict_total_model)],
    mongo: Annotated[MongoService, Depends(get_mongo_service)]
    ):

    preds = model_service.predict(
        radiant_heroes=predict_data.radiant_picks,
        dire_heroes=predict_data.dire_picks
    )
    stats = mongo.get_total_stats(
        radiant_team_id=predict_data.radiant_team_id,
        dire_team_id=predict_data.dire_team_id,
        radaint_heroes=predict_data.radiant_picks,
        dire_heroes=predict_data.dire_picks
    )
    # heroes_winrate = mongo.get_hero_stats(
    #     radiant_team_id=predict_data.radiant_team_id,
    #     dire_team_id=predict_data.dire_team_id,
    #     radaint_heroes=predict_data.radiant_picks,
    #     dire_heroes=predict_data.dire_picks
    # )

    total_preds: np.array = (stats * 0.45 + preds * 0.55).astype(np.int16)[0]

    # preds: np.array = (stats * 0.55 + preds.flatten() * 0.45).astype(np.int16)

    lower = 42.5
    upper = 54.5
    diff = upper - lower

    total_over_coefs = np.linspace(*predict_data.total_over_coefs, int(diff+1))
    total_less_coefs = np.linspace(*predict_data.total_less_coefs, int(diff+1))
    totals = np.linspace(lower, upper, int(diff+1))

    lower_threshold = 3
    upper_threshold = 4

    total_preds = total_preds + predict_data.bias - 0.5

    bank = predict_data.bank
    bet = bank * predict_data.bet_coef

    total_over = total_preds - lower_threshold
    total_less = total_preds + upper_threshold
    total_over = 54.5 if total_over > 54.5 else total_over
    total_less = 42.5 if total_less < 42.5 else total_less
    if 42.5 <= total_over <= 54.5:
        ind = np.where(totals == total_over)[0][0]
        coef_over = total_over_coefs[ind]
    else:
        coef_over = 0
    if 42.5 <= total_less <= 54.5:
        ind = np.where(totals == total_less)[0][0]
        coef_less = total_less_coefs[ind]
    else:
        coef_less = 0

    if coef_over > coef_less:
        return {
            "total_over": float(total_over),
            "ml_predict": float(preds),
            "stats": float(stats[0]),
            "sum_predict": float(total_preds),
            "total_less": float(total_less),
            # "heroes_stats": heroes_winrate,
            "bet": f"{bet:.2f} on total over {total_over}, coef {coef_over:.2f}",
            "coef_less": float(coef_less)
        }
    else:
        return {
            "total_over": float(total_over),
            "ml_predict": float(preds),
            "stats": float(stats[0]),
            "sum_predict": float(total_preds),
            "total_less": float(total_less),
            # "heroes_stats": heroes_winrate,
            "bet": f"{bet:.2f} on total less {total_less}, coef {coef_less:.2f}",
            "coef_over": float(coef_over)
        }

