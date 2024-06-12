import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as scp


def show_team_stats(data: dict) -> plt.figure:
    """Return a figure with 2 plots: radiant win and radiant lose."""

    plt.style.use('ggplot')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot radiant lose
    ax.scatter(
        data["radiant_team_lose_duration"],
        data["radiant_team_lose_total"],
        color=mcolors.CSS4_COLORS["mediumturquoise"],
        label="Radiant team lose"
    )
    ax.scatter(
        data["dire_team_win_duration"],
        data["dire_team_win_total"],
        color=mcolors.CSS4_COLORS["indianred"],
        label="Dire team win"
    )

    # total mean if radiant lose
    rldw_total = round((np.array(data["radiant_team_lose_total"] + data["dire_team_win_total"])).mean(), 2)
    ax.plot(
        np.linspace(0, 100, 10),
        [rldw_total for _ in range(10)],
        "-.k", label=f"Total mean {rldw_total}"
    )
    # duration mean if radiant lose
    ax.plot(
        [round((np.array(data["radiant_team_lose_duration"] + data["dire_team_win_duration"])).mean(), 2) for _ in range(10)],
        np.linspace(0, 100, 10), "-.b"
    )

    ax.legend()
    ax.set_xlabel("Duration")
    ax.set_ylabel("Total")
    ax.set_ylim(10, 80)
    ax.set_xlim(10, 80)
    ax.set_title("Radiant lose")


    # plot radiant win
    ax2.scatter(
        data["radiant_team_win_duration"],
        data["radiant_team_win_total"],
        color=mcolors.CSS4_COLORS["mediumturquoise"],
        label="Radiant team win"
    )
    ax2.scatter(
        data["dire_team_lose_duration"],
        data["dire_team_lose_total"],
        color=mcolors.CSS4_COLORS["indianred"],
        label="Dire team lose"
    )

    # total mean if radiant win
    rwdl_total = round((np.array(data["radiant_team_win_total"] + data["dire_team_lose_total"])).mean(), 2)
    ax2.plot(
        np.linspace(0, 100, 10),
        [rwdl_total for _ in range(10)], "-.k",
        label=f"Total mean {rwdl_total}"
    )
    
    # duration mean if radiant win
    ax2.plot(
        [round((np.array(data["radiant_team_win_duration"] + data["dire_team_lose_duration"])).mean(), 2) for _ in range(10)],
        np.linspace(0, 100, 10), "-.b"
    )

    ax2.legend()
    ax2.set_xlabel("Duration")
    ax2.set_ylabel("Total")
    ax2.set_ylim(10, 80)
    ax2.set_xlim(10, 80)
    ax2.set_title("Radiant win")
    
    return fig


def show_stats(data: dict, teams: tuple) -> plt.figure:
    """Return figure with 3 plots:
        - win odds,
        - total distribution,
        - duration distribution.
    """

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

    fig, ax = plt.subplots(3, layout='constrained', figsize=(8, 11))

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


    means_dur = [data["duration"]["radiant_team"]["mu"], data["duration"]["dire_team"]["mu"]]
    std_dur_values = [data["duration"]["radiant_team"]["sigma"], data["duration"]["dire_team"]["sigma"]]

    maxs_dur = []

    for mu, std, style in zip(means_dur, std_dur_values, styles):
        probabilities = scp.stats.norm.pdf(domain, mu, std)
        maxs_dur.append(probabilities.max()/1.65)
        ax[2].plot(domain, probabilities, style, label=f"dur $\mu={mu:.2f}$\ndur $\sigma={std:.2f}$\n")
    
    ax[2].plot(
        [data["duration"]["ridge"] for _ in range(10)],
        np.linspace(0, max(maxs_dur)*1.75, 10),
        "-.k", label=f"Ridge predict {data['duration']['ridge']}"
    )
    ax[2].plot(
        [data["duration"]["sum_dur"] for _ in range(10)],
        np.linspace(0, max(maxs_dur)*1.75, 10),
        "-b", label=f"Sum duration {data['duration']['sum_dur']}"
    )

    ax[2].plot(
        [34 for _ in range(10)],
        np.linspace(0, max(maxs_dur)*1.75, 10),
        "-.y",
        label=f"Dur low threshold"
    )
    ax[2].plot(
        [40 for _ in range(10)],
        np.linspace(0, max(maxs_dur)*1.75, 10),
        "-.y",
        label=f"Dur high threshold")

    y3 = np.array([max(maxs_dur)*1.75 for _ in range(2)])
    y4 = np.array([0, 0])

    ax[2].fill_between(
        np.array([34, 40]),
        y3,
        y4,
        where=(y3 > y4), color='C1', alpha=0.3    
    )

    ax[2].legend()
    ax[2].set_xlabel("Duration")
    ax[2].set_ylabel("Probability")

    return fig
