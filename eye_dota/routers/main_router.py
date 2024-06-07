# import numpy as np
import polars as pl
from pydantic import ValidationError
from typing import Annotated
from fastapi import APIRouter, Depends
from schemas import PredictTotal, Match
from services import (
    PredictTotalModel,
    MongoService,
    AiohttpService,
    RidgeDurationModel
)
from dependencies import (
    get_predict_total_model,
    get_mongo_service,
    get_aiohttp_service,
    get_rigde_duration
)


main = APIRouter(
    prefix="/main",
    tags=["Main"]
)


@main.post("/predict_total")
async def predict_total(
    predict_data: PredictTotal,
    model_service: Annotated[PredictTotalModel, Depends(get_predict_total_model)],
    mongo: Annotated[MongoService, Depends(get_mongo_service)],
    ridge: Annotated[RidgeDurationModel, Depends(get_rigde_duration)]
):

    model_predict: int = model_service.predict(
        radiant_heroes=predict_data.radiant_picks,
        dire_heroes=predict_data.dire_picks
    )

    ridge_predict: int = ridge.predict(
        radiant_heroes=predict_data.radiant_picks,
        dire_heroes=predict_data.dire_picks
    )

    total_and_dur_data = mongo.get_teams_total_mean_and_std(
        predict_data.radiant_team_id,
        predict_data.dire_team_id,
        model_predict=model_predict,
        ridge_predict=ridge_predict
    )

    wilson_data = mongo.get_wilson_odds(
        predict_data.radiant_team_id,
        predict_data.dire_team_id,
        predict_data.radiant_picks,
        predict_data.dire_picks
    )

    result = {
        "total_over": total_and_dur_data["total_over"],
        "ml_predict": total_and_dur_data["model_predict"],
        "sum_predict": total_and_dur_data["total_preds"],
        "total_less": total_and_dur_data["total_less"],
        "radiant_team": {
            "mu": total_and_dur_data["rt_total_mean"],
            "sigma": total_and_dur_data["rt_total_std"]
        },
        "dire_team": {
            "mu": total_and_dur_data["dt_total_mean"],
            "sigma": total_and_dur_data["dt_total_std"]
        },
        "duration": {
            "sum_dur": total_and_dur_data["dur_preds"],
            "ridge": total_and_dur_data["ridge_predict"],
            "radiant_team": {
                "mu": total_and_dur_data["rt_dur_mean"],
                "sigma": total_and_dur_data["rt_dur_std"],
            },
            "dire_team": {
                "mu": total_and_dur_data["dt_dur_mean"],
                "sigma": total_and_dur_data["dt_dur_std"],
            }
        },
        "wilson_data": wilson_data
    }

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


@main.get("/teams_data")
async def update_data(
    radiant_team_id: int,
    dire_team_id: int,
    last_matches: int,
    mongo: Annotated[MongoService, Depends(get_mongo_service)]
):
    return mongo.get_team_stats_for_n_last_matches(
        radiant_team_id=radiant_team_id,
        dire_team_id=dire_team_id,
        last_matches=last_matches
    )
