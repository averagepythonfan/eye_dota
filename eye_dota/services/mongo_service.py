from pymongo import MongoClient
import polars as pl
import numpy as np


class MongoService:

    def __init__(self, uri: str) -> None:
        self.client = MongoClient(uri)


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