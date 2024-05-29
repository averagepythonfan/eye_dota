from datetime import datetime, timedelta, timezone
from typing import Tuple
from pymongo import MongoClient
import polars as pl
import numpy as np


class MongoService:

    def __init__(self, uri: str) -> None:
        self.client = MongoClient(uri)

    def get_dates(self) -> Tuple[str, str]:
        cur = self.client.data.matches.find({}, {"start_match":1}).sort({"match_id": -1}).limit(1)
        last_match = list(cur)[0]["start_match"]
        last_time_str = last_match.strftime("%Y-%m-%dT%H:%M:%S")
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        return last_time_str, now_str


    def get_total_stats(self,
                        radiant_team_id: int,
                        dire_team_id: int,
                        radaint_heroes: list[int],
                        dire_heroes: list[int]):
        
        cur = self.client.data.matches.find(
            {"$or": [
                {
                    "radiant_team_id": radiant_team_id, "radiant_picks": {"$in": radaint_heroes}
                },
                {
                    "dire_team_id": dire_team_id,  "dire_picks": {"$in": dire_heroes}
                }
            ]},
            {"match_id": 1, "radiant_team_id": 1, "dire_team_id": 1, "radiant_score": 1, "dire_score": 1 , "radiant_win": 1}
        )
        data = list(cur)
        teams_data = pl.DataFrame(data)

        results = []
        
        with pl.SQLContext(frames={
            "radiant": teams_data.sort("match_id", descending=True).filter(pl.col("radiant_team_id").eq(radiant_team_id)).limit(15),
            "dire": teams_data.sort("match_id", descending=True).filter(pl.col("dire_team_id").eq(dire_team_id)).limit(15)
        }) as ctx:
            res = ctx.execute(f"""
                SELECT
                    sum(radiant_score + dire_score) / count(*) as total_mean 
                FROM radiant
                WHERE
                    radiant_team_id = {radiant_team_id}

                UNION
        
                SELECT
                    sum(radiant_score + dire_score) / count(*) as total_mean 
                FROM dire
                WHERE
                    dire_team_id = {dire_team_id}
            """)
            results.append(res.collect().mean().to_numpy().flatten()[0])
        return np.array(results).astype(np.int16)


    def get_hero_stats(self,
                       radiant_team_id: int,
                       dire_team_id: int,
                       radaint_heroes: list[int],
                       dire_heroes: list[int]) -> list[dict]:
        
        heroes_data = {el["id"]: el["localized_name"] for el in self.client.data.heroes.find({}, {"id": 1, "_id": 0, "localized_name": 1})}

        cur = self.client.data.matches.find(
            {"$or": [
                {
                    "radiant_team_id": radiant_team_id, "radiant_picks": {"$in": radaint_heroes}
                },
                {
                    "dire_team_id": dire_team_id,  "dire_picks": {"$in": dire_heroes}
                }
            ]},
            {
                "radiant_team_id": 1, "dire_team_id": 1, "radiant_picks": 1, "dire_picks": 1, "radiant_win": 1
            }
        )

        stats = pl.DataFrame(list(cur))

        radiant_hero_stats = pl.DataFrame(schema=[("hero", pl.String), ("winrate", pl.Float32), ("count", pl.Int32)])
        dire_hero_stats = pl.DataFrame(schema=[("hero", pl.String), ("winrate", pl.Float32), ("count", pl.Int32)])

        for team, data in {
            "radiant": {
                "radiant_team_id": radiant_team_id,
                "radiant_picks": radaint_heroes
            },
            "dire": {
                "dire_team_id": dire_team_id,
                "dire_picks": dire_heroes
            }
        }.items():
            if team == "radiant":
                team_id = data["radiant_team_id"]
                picks = data["radiant_picks"]
                for hero in picks:
                    hero_stat = stats.filter([
                        pl.col("radiant_team_id").eq(team_id),
                        pl.col("radiant_picks").list.contains(hero)
                    ]).select([
                        pl.lit(heroes_data[hero]).alias("hero"),
                        pl.col("radiant_win").mean().round(2).cast(pl.Float32).alias("winrate"),
                        pl.col("radiant_win").count().cast(pl.Int32).alias("count")
                    ])
                    radiant_hero_stats.extend(hero_stat)
            elif team == "dire":
                team_id = data["dire_team_id"]
                picks = data["dire_picks"]
                for hero in picks:
                    hero_stat = stats.filter([
                        pl.col("dire_team_id").eq(team_id),
                        pl.col("dire_picks").list.contains(hero)
                    ]).select([
                        pl.lit(heroes_data[hero]).alias("hero"),
                        pl.col("radiant_win").not_().mean().round(2).cast(pl.Float32).alias("winrate"),
                        pl.col("radiant_win").count().cast(pl.Int32).alias("count")
                    ])
                    dire_hero_stats.extend(hero_stat)
                
        radiant_hero_stats = pl.concat([
            radiant_hero_stats,
            radiant_hero_stats.select([
                pl.lit("Over All").alias("hero"),
                pl.col("winrate").mean().round(2),
                pl.col("count").sum()
            ])
        ])

        dire_hero_stats = pl.concat([
            dire_hero_stats,
            dire_hero_stats.select([
                pl.lit("Over All").alias("hero"),
                pl.col("winrate").mean().round(2),
                pl.col("count").sum()
            ])
        ])

        return {
            "radiant": [
                {
                    "hero": el["hero"],
                    "wr": round(el["winrate"], 2),
                    "count": el["count"]
                } for el in radiant_hero_stats.to_dicts()
            ],
            "dire": [
                {
                    "hero": el["hero"],
                    "wr": round(el["winrate"], 2),
                    "count": el["count"]
                } for el in dire_hero_stats.to_dicts()
            ]
        }


    def _winrate_wilson(self, matches: int, winrate: int) -> int:
        if matches == 0:
            return 0
        winrate = winrate / 100 if isinstance(winrate, int) else round(winrate, 2)
        wilson_coef = 1.96
        result = (
            winrate + wilson_coef * wilson_coef / (2 * matches) - wilson_coef * np.sqrt(
                (winrate * (1 - winrate) + wilson_coef * wilson_coef / (4 * matches)) / matches
            )
        ) / (
            1 + wilson_coef * wilson_coef / matches
        )
        return int(result * 100)


    def _team_hero_stats(self, team_id: int, picks: list[int]) -> int:

        one_year_ago = datetime.now() - timedelta(days=365)

        cur = self.client.data.matches.find({"$or": [
            {"radiant_team_id": team_id, "radiant_picks": {"$in": picks}, "start_match": {"$gt": one_year_ago}},
            {"dire_team_id": team_id, "dire_picks": {"$in": picks}, "start_match": {"$gt": one_year_ago}}
        ]}, {"radiant_team_id": 1, "dire_team_id": 1, "radiant_picks": 1, "dire_picks": 1, "radiant_win": 1})

        team_df = pl.DataFrame(list(cur))

        picks_df = team_df.filter(pl.col("radiant_team_id").eq(team_id)).select([
            pl.col("radiant_picks").alias("picks"),
            pl.col("radiant_win").cast(pl.Int8).alias("win")
        ]).vstack(team_df.filter(pl.col("dire_team_id").eq(team_id)).select([
            pl.col("dire_picks").alias("picks"),
            pl.col("radiant_win").not_().cast(pl.Int8).alias("win")
        ]))

        wilson = 0
        for hero in picks:
            hero_values = picks_df.filter(pl.col("picks").list.contains(hero)).select([
                pl.col("win").count().cast(pl.Int32).alias("count"),
                pl.col("win").mean().round(2).cast(pl.Float32).alias("winrate")
            ]).fill_null(0).to_dicts().pop().values()
            wilson += self._winrate_wilson(*hero_values)
        return wilson


    def get_patch_stats(self, all_heroes: list[int], current_patch: str = "7.36"):

        assert len(all_heroes) == 10
        
        cur = self.client.data.matches.find({"$or": [
            {"patch": current_patch, "radiant_picks": {"$in": all_heroes}},
            {"patch": current_patch, "dire_picks": {"$in": all_heroes}}
        ]}, {"radiant_team_id": 1, "dire_team_id": 1, "radiant_picks": 1, "dire_picks": 1, "radiant_win": 1})

        heroes_df = pl.DataFrame(list(cur))

        picks_df = heroes_df.select([
            pl.col("radiant_picks").alias("picks"),
            pl.col("radiant_win").cast(pl.Int8).alias("win")
        ]).vstack(heroes_df.select([
            pl.col("dire_picks").alias("picks"),
            pl.col("radiant_win").not_().cast(pl.Int8).alias("win")
        ]))

        heroes_list = []
        
        for hero in all_heroes:
            hero_values = picks_df.filter(pl.col("picks").list.contains(hero)).select([
                pl.col("win").count().cast(pl.Int32).alias("count"),
                pl.col("win").mean().round(2).cast(pl.Float32).alias("winrate")
            ]).fill_null(0).to_dicts().pop().values()
            heroes_list.append(self._winrate_wilson(*hero_values))
        return sum(heroes_list[:5]), sum(heroes_list[5:])


    def _wilson_odds(self, radiant_wilson: int, dire_wilson: int):
        
        if radiant_wilson == dire_wilson:
            return 0.5, 0.5

        wilson_weight = np.log(abs(radiant_wilson - dire_wilson))

        radiant_value = np.sqrt((radiant_wilson / (radiant_wilson + dire_wilson)) ** wilson_weight)
        dire_value = np.sqrt((dire_wilson / (radiant_wilson + dire_wilson)) ** wilson_weight)
        
        radiant_odds = round(radiant_value / (radiant_value + dire_value), 2)
        dire_odds = round(1 - radiant_odds, 2)
        return radiant_odds, dire_odds

    def _get_heroes_advantages(self, radiant_heroes: list[int], dire_heroes: list[int], on: str = "current_patch"):
        cur = self.client.stats.heroes.find({"hero_id": {"$in": radiant_heroes+dire_heroes}})
        df = list(cur)
        radiant_adv, dire_adv = 0, 0
        
        for hero in df:
            if hero["hero_id"] in radiant_heroes:
                for wilson_list in hero[on]:
                    if wilson_list["hero_id"] in dire_heroes:
                        radiant_adv += wilson_list["wilson_adv"]
            if hero["hero_id"] in dire_heroes:
                for wilson_list in hero[on]:
                    if wilson_list["hero_id"] in radiant_heroes:
                        dire_adv += wilson_list["wilson_adv"]

        return self._wilson_odds(radiant_adv, dire_adv)


    def get_wilson_odds(self,
                        radiant_team_id: int,
                        dire_team_id: int,
                        radiant_heroes: list[int],
                        dire_heroes: list[int]):
        radiant_wilson = self._team_hero_stats(team_id=radiant_team_id, picks=radiant_heroes)
        dire_wilson = self._team_hero_stats(team_id=dire_team_id, picks=dire_heroes)

        teams_wilson = self._wilson_odds(radiant_wilson, dire_wilson)
        patch_adv_wilson = self._get_heroes_advantages(radiant_heroes, dire_heroes)
        last_year_adv_wilson = self._get_heroes_advantages(radiant_heroes, dire_heroes, on="last_year")

        patch_data = self.get_patch_stats(radiant_heroes+dire_heroes)
        patch_wilson = self._wilson_odds(*patch_data)

        return {
            "patch_data_wilson_odds": patch_wilson,
            "advantages_patch": patch_adv_wilson,
            "team_data_wilson_odds": teams_wilson,
            "advantages_year": last_year_adv_wilson
        }


    def reinit_patch_advantages(self):
        current_patch = "7.36"
        cur = self.client.data.matches.find({"patch": current_patch}, {
            "radiant_picks": 1,
            "dire_picks": 1,
            "radiant_win": 1    
        })

        df = pl.DataFrame(list(cur))

        heroes = {
            el["localized_name"]: el["id"] for el in self.client.data.heroes.find({}, {"id": 1, "_id": 0, "localized_name": 1})
        }

        heroes_id = list(heroes.values())

        heroes_data_adv = []
        for hero_id in heroes_id:
            hero_df = df.filter([
                pl.col("radiant_picks").list.contains(hero_id),
            ]).select([
                pl.col("dire_picks").alias("picks"),
                pl.col("radiant_win").cast(pl.Int8).alias("win")
            ]).vstack(
                df.filter([
                    pl.col("dire_picks").list.contains(hero_id)
                ]).select([
                    pl.col("radiant_picks").alias("picks"),
                    pl.col("radiant_win").not_().cast(pl.Int8).alias("win")
                ])
            )

            stats = dict()
            for el, win in zip(hero_df["picks"], hero_df["win"]):
                heroes = list(el)
                for h in heroes:
                    if stats.get(h) == None:
                        stats[h] = [win]
                    else:
                        stats[h].append(win)

            for hero in stats:
                l = stats[hero]
                stats[hero] = self._winrate_wilson(len(l), float(np.mean(l)))

            mongo_stats = []
            for h_id, wilson in stats.items():
                mongo_stats.append({
                    "hero_id": h_id,
                    "wilson_adv": wilson
                })
            
            heroes_data_adv.append({
                "hero_id": hero_id,
                "current_patch": mongo_stats
            })
        
        updated_count = 0

        for data in heroes_data_adv:
            res = self.client.stats.heroes.update_one(
                {"hero_id": data["hero_id"]},
                {"$set": {"current_patch": data["current_patch"]}}
            )
            if res.acknowledged:
                updated_count += 1
        return updated_count


    def get_teams_total_mean_and_std(self, radiant_team_id: int, dire_team_id: int) -> Tuple[float, float, float, float]:

        cur = self.client.data.matches.find({"$or": [
            {"radiant_team_id": radiant_team_id},
            {"dire_team_id": radiant_team_id}
        ]}, {"radiant_score": 1, "dire_score": 1}).sort({"match_id": -1}).limit(5)

        rt = pl.DataFrame(list(cur))
        rt_totals = rt.select(pl.col("radiant_score").add(pl.col("dire_score"))).to_numpy().flatten()

        cur = self.client.data.matches.find({"$or": [
            {"radiant_team_id": dire_team_id},
            {"dire_team_id": dire_team_id}
        ]}, {"radiant_score": 1, "dire_score": 1}).sort({"match_id": -1}).limit(5)

        dt = pl.DataFrame(list(cur))
        dt_totals = dt.select(pl.col("radiant_score").add(pl.col("dire_score"))).to_numpy().flatten()

        return rt_totals.mean(), rt_totals.std(), dt_totals.mean(), dt_totals.std()
