import streamlit as st
from heroes import heroes, teams
import requests
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


st.title("DotA MainPredictor App")


st.text("Let's predict...")


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


bank: int = st.number_input("Bank value", min_value=100)
bet_coef: float = st.number_input("Bank value", min_value=0.1, max_value=0.35, value=0.2)
bias: int = st.number_input("Bias value, default is -1", value=-1)

total_over_low: float = st.number_input("Total Over Low Threshold", value=1.48)
total_over_up: float = st.number_input("Total Over Up Threshold", value=2.72)

total_less_low: float = st.number_input("Total Less Low Threshold", value=2.72)
total_less_up: float = st.number_input("Total Less Up Threshold", value=1.42)


# @st.cache_data
def preditct(predict_data: PredictTotal):

    response = requests.post("http://eye_dota:9090/main/predict_total", json=predict_data.model_dump())
    if response.status_code == 200:
        return response.json()
    else:
        return {"fail": response.json()}



if len(rhero) == 5 and len(dhero) == 5:
    if st.button("Predict"):
        st.write("Your predict is...")
        st.json(
            preditct(
                predict_data=PredictTotal(
                    radiant_team_id=teams[radiant_team[0]],
                    dire_team_id=teams[dire_team[0]],
                    radiant_picks=radiant_hero_id,
                    dire_picks=dire_hero_id,
                    bank=bank,
                    bet_coef=bet_coef,
                    bias=bias,
                    total_over_coefs=(total_over_low, total_over_up),
                    total_less_coefs=(total_less_low, total_less_up)
                )
            )
        )
