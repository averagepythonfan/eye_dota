# import numpy as np
import polars as pl
from pydantic import ValidationError
from typing import Annotated
from fastapi import APIRouter, Depends
from schemas import PredictTotal, Match
from services import PredictTotalModel, MongoService, AiohttpService
from dependencies import get_predict_total_model, get_mongo_service, get_aiohttp_service


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

    model_predict = model_service.predict(
        radiant_heroes=predict_data.radiant_picks,
        dire_heroes=predict_data.dire_picks
    )

    rt_total_mean, rt_total_std, dt_total_mean, dt_total_std = mongo.get_teams_total_mean_and_std(
        predict_data.radiant_team_id,
        predict_data.dire_team_id
    )


    wilson_data = mongo.get_wilson_odds(
        predict_data.radiant_team_id,
        predict_data.dire_team_id,
        predict_data.radiant_picks,
        predict_data.dire_picks
    )

    std_values = [rt_total_std, dt_total_std]

    stats_coef = 0.7
    model_coef = 1 - stats_coef

    weights = [round(stats_coef - el / sum(std_values) * stats_coef, 2) for el in std_values] + [model_coef]
    total_preds = rt_total_mean * weights[0] + dt_total_mean * weights[1] + model_predict * weights[2]
    total_preds = int(total_preds)

    lower_threshold = 3
    upper_threshold = 4

    total_preds = total_preds + predict_data.bias - 0.5

    total_over = total_preds - lower_threshold
    total_less = total_preds + upper_threshold

    result = {
        "total_over": round(float(total_over), 2),
        "ml_predict": round(float(model_predict), 2),
        "sum_predict": round(float(total_preds), 2),
        "total_less": round(float(total_less), 2),
        "radiant_team": {
            "mu": round(rt_total_mean, 2),
            "sigma": round(rt_total_std, 2)
        },
        "dire_team": {
            "mu": round(dt_total_mean,2),
            "sigma": round(dt_total_std, 2)
        },
        "wilson_data": wilson_data
    }
    
    print(result)

    return result


@main.get("/update_data")
async def update_data(
    mongo: Annotated[MongoService, Depends(get_mongo_service)],
    aio: Annotated[AiohttpService, Depends(get_aiohttp_service)]
):
    last, end = mongo.get_dates()
    data_json = await aio.get_matches_update_request(last_date=last, end_date=end)

    data_list = list()
    for el in data_json:
        data = tuple(el.values())
        data_list.append(data)
    cols = data_json[0].keys()
    data_T = list(zip(*data_list))

    polars_schema = [
        ("match_id", pl.Int64),
        ("start_match", pl.Datetime(time_unit='ms')),
        ("league_id", pl.Int64),
        ("tier", pl.Boolean),
        ("radiant_team_id", pl.Int64),
        ("dire_team_id", pl.Int64),
        ("radiant_win", pl.Boolean),
        ("radiant_score", pl.Int32),
        ("dire_score", pl.Int32),
        ("duration", pl.Int32),
        ("patch", pl.String),
        ("gold", pl.List(inner=pl.Int32)),
        ("xp", pl.List(inner=pl.Int32)),
        ("picks", pl.List(inner=pl.Int32)),
        ("players", pl.List(inner=pl.Utf8)),
    ]

    df = pl.DataFrame(data={col:data for col, data in zip(cols, data_T)}, schema=polars_schema)

    for el in df['picks']:
        if el.len() != 10:
            df = df.filter(pl.col("picks") != el.reshape((1, el.shape[0])))
            raise ValidationError()
    for el in df['players']:
        if el.len() != 10:
            error = el
            raise ValidationError()
    for el in (df.null_count() == 0):
        if el[0] is not True:
            raise ValidationError()
    if df.is_empty():
        raise ValidationError()

    df = df.with_columns(
            pl.col("players").cast(pl.List(pl.Int64))
    ).with_columns([
            pl.col('picks').list.slice(0, 5).alias("radiant_picks"),
            pl.col('picks').list.slice(5, 10).alias("dire_picks"),
            pl.col("players").list.slice(0, 5).alias("radiant_players"),
            pl.col("players").list.slice(5, 10).alias("dire_players")
    ]).drop("picks").drop("players")

    not_inserted = 0
    for el in df.to_dicts():
        try:
            match_data = Match(**el)
        except:
            not_inserted += 1
            continue
        res = mongo.client.data.matches.insert_one(match_data.model_dump(by_alias=True))
        if not res.acknowledged:
            not_inserted += 1

    update_count = mongo.reinit_patch_advantages()

    return {
        "matches_count": len(data_json),
        "not_inserted": not_inserted,
        "adv_update": update_count
    }