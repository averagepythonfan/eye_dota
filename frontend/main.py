from typing import Dict
import streamlit as st
from heroes import heroes, teams
import requests
from pydantic import BaseModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy as scp


class PredictTotal(BaseModel):
    radiant_team_id: int
    dire_team_id: int
    radiant_picks: list[int]
    dire_picks: list[int]
    bias: int


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


bias: int = st.number_input("Bias value, default is -1", value=-1)


# @st.cache_data
def preditct(predict_data: PredictTotal):

    response = requests.post("http://eye_dota:9090/main/predict_total", json=predict_data.model_dump())
    if response.status_code == 200:
        return response.json()
    else:
        return {"fail": response.json()}


def show_stats(data: Dict) -> plt.figure:
    teams = (radiant_team[0], dire_team[0])

    colors = [
        mcolors.CSS4_COLORS["lightcoral"],
        mcolors.CSS4_COLORS["paleturquoise"],
        mcolors.CSS4_COLORS["moccasin"],
        mcolors.CSS4_COLORS["yellowgreen"]
    ]

    domain = np.linspace(10, 80, 100) 

    means = [data["radiant_team"]["mu"], data["dire_team"]["mu"]]
    std_values = [data["radiant_team"]["sigma"], data["dire_team"]["sigma"]]

    model_predict = data["ml_predict"]
    predict = data["sum_predict"]

    styles = ["c", "r"]
    maxs = []


    x = np.arange(len(teams))
    width = 0.15  
    multiplier = 0

    fig, ax = plt.subplots(2, layout='constrained', figsize=(8, 8))

    for attribute, measurement in data["wilson_data"].items():
        offset = width * multiplier
        rects = ax[0].bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
        ax[0].bar_label(rects, padding=3)
        multiplier += 1

    ax[0].set_ylabel('Odds div by 100')
    ax[0].set_title('Win odds by stats')
    ax[0].set_xticks(x + width + 0.1, teams)
    ax[0].legend(loc='upper left', ncols=3)
    ax[0].set_ylim(0, 1)

    for mu, std, style in zip(means, std_values, styles):
        probabilities = scp.stats.norm.pdf(domain, mu, std)
        maxs.append(probabilities.max()/1.65)
        ax[1].plot(domain, probabilities, style, label=f"$\mu={mu:.2f}$\n$\sigma={std:.2f}$\n")

    # ax[1].plot([means[0] for _ in range(10)], np.linspace(0, maxs[0]*1.65, 10), "-c")
    # ax[1].plot([means[1] for _ in range(10)], np.linspace(0, maxs[1]*1.65, 10), "-r")

    ax[1].plot([model_predict for _ in range(10)], np.linspace(0, max(maxs)*1.75, 10), "-.k", label="Model predict")
    ax[1].plot([predict for _ in range(10)], np.linspace(0, max(maxs)*1.75, 10), "-b", label=f"Sum predict {predict}")

    ax[1].plot(
        [data["total_over"] for _ in range(10)],
        np.linspace(0, max(maxs)*1.75, 10),
        "-.b",
        label=f"Total over: {data['total_over']}"
    )
    ax[1].plot(
        [data["total_less"] for _ in range(10)],
        np.linspace(0, max(maxs)*1.75, 10),
        "-.b",
        label=f"Total less: {data['total_less']}")

    y1 = np.array([max(maxs)*1.75 for _ in range(3)])
    y2 = np.array([0, 0, 0])

    ax[1].fill_between(
        np.array([data["total_over"], data["sum_predict"], data["total_less"]]),
        y1,
        y2,
        where=(y1 > y2), color='C0', alpha=0.3    
    )

    ax[1].legend()
    ax[1].set_xlabel("Total")
    ax[1].set_ylabel("Probability")

    return fig


if len(rhero) == 5 and len(dhero) == 5:
    if st.button("Predict"):
        st.pyplot(show_stats(
            preditct(
                predict_data=PredictTotal(
                    radiant_team_id=teams[radiant_team[0]],
                    dire_team_id=teams[dire_team[0]],
                    radiant_picks=radiant_hero_id,
                    dire_picks=dire_hero_id,
                    bias=bias,
                )
            )
        )
        )


if st.button("Update data"):
    response = requests.get("http://eye_dota:9090/main/update_data")
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.json({"fail": response.json()})
