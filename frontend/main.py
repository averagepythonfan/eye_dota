import streamlit as st
from func import show_team_stats, show_stats
from heroes import heroes, teams
import requests
from pydantic import BaseModel


class PredictTotal(BaseModel):
    radiant_team_id: int
    dire_team_id: int
    radiant_picks: list[int]
    dire_picks: list[int]
    bias: int


st.title("DotA MainPredictor App")


# teams
radiant_team: list = st.multiselect(
    "Radiant Team: ",
    teams,
    max_selections=1
)

dire_team: list = st.multiselect(
    "Dire Team: ",
    teams,
    max_selections=1
)


# heroes
rhero = st.multiselect(
    "Team Radiant heroes:",
    heroes,
    max_selections=5
    )

dhero = st.multiselect(
    "Team Dire heroes:",
    {key: heroes[key] for key in heroes if key not in rhero},
    max_selections=5
    )


radiant_hero_id = list()
for el in rhero:
    radiant_hero_id.append(heroes[el])

dire_hero_id = list()
for el in dhero:
    dire_hero_id.append(heroes[el])


bias: int = st.number_input("Bias value, default is -1", value=-1)
last_matches: int = st.number_input("Last matches for team stats", value=15)


def preditct(predict_data: PredictTotal):

    response = requests.post("http://eye_dota:9090/main/predict_total", json=predict_data.model_dump())
    if response.status_code == 200:
        return response.json()
    else:
        return {"fail": response.json()}


if len(rhero) == 5 and len(dhero) == 5:
    if st.button("Predict"):
        st.pyplot(show_stats(
            data=preditct(
                predict_data=PredictTotal(
                    radiant_team_id=teams[radiant_team[0]],
                    dire_team_id=teams[dire_team[0]],
                    radiant_picks=radiant_hero_id,
                    dire_picks=dire_hero_id,
                    bias=bias,
                )
            ),
            teams=(radiant_team[0], dire_team[0])
        ))


if st.button("Update data"):
    response = requests.get("http://eye_dota:9090/main/update_data")
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.json({"fail": response.json()})


if radiant_team and dire_team:
    if st.button("Team stats"):
        resp = requests.get(
            "http://eye_dota:9090/main/teams_data",
            params={
                "radiant_team_id": teams[radiant_team[0]],
                "dire_team_id": teams[dire_team[0]],
                "last_matches": last_matches
                }
        )
        if resp.status_code == 200:
            st.pyplot(
                show_team_stats(resp.json())
            )
