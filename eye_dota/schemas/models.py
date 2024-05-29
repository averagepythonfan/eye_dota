from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from bson import ObjectId


class PredictTotal(BaseModel):
    radiant_team_id: int
    dire_team_id: int
    radiant_picks: list[int]
    dire_picks: list[int]
    bias: int


class Collections(Enum):
    matches = "matches"
    time_series = "time_series"


class Meta(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    collection: Optional[Collections] = Field(default=None, exclude=True, repr=False)

    id: ObjectId = Field(default_factory=lambda:ObjectId(), frozen=True, alias="_id")
    match_id: int


class Match(Meta):
    collection: Collections = Field(default=Collections.matches, exclude=True, repr=False)

    start_match: datetime = Field(default_factory=lambda x: datetime.fromtimestamp(x), frozen=True)
    league_id: int
    tier: bool
    radiant_team_id: int
    dire_team_id: int
    radiant_players: list[int]
    dire_players: list[int]
    radiant_picks: list[int]
    dire_picks: list[int]
    radiant_score: int
    dire_score: int
    duration: int
    radiant_win: bool
    patch: str
    gold: list[int]
    xp: list[int]
