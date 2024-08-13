"""
This script recreates fig 2 of the manuscript.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table

unfed_female_tsetse = {
    "1": [21.5, 6.7, 6.4, 6.4],
    "2": [23.2, 6.8, 5.7, 6.1],
    "3": [20.3, 5.8, 6.0, 6.0],
    "4": [20.4, 5.8, 5.8, 5.8],
    "5": [18.6, 5.2, 5.0, 5.0],
    "6": [16.4, 4.5, 4.3, 4.3],
    "7": [19.3, 5.6, 5.5, 5.5],
    "8": [19.4, 6.8, 6.3, 6.4],
    "9": [16.7, 5.1, 5.1, 5.3],
    "10": [12.1, 2.9, 2.9, 2.9],
}

unfed_female_tsetsedf = pd.DataFrame(
    unfed_female_tsetse, index=["0 h", "24 h", "72 h", "120 h"]
)
unfed_female_tsetsedf["time test"] = ["0 h", "24 h", "72 h", "120 h"]

blood_female_tsetse = {
    "1": [60.4, 35.8, 26.7, 26.4],
    "2": [55.7, 31.7, 24.1, 23.9],
    "3": [67.5, 44.1, 31.5, 31.0],
    "4": [52.7, 30.5, 21.1, 20.8],
    "5": [60.1, 36.6, 24.7, 24.3],
    "6": [48.6, 25.8, 19.7, 19.4],
    "7": [59.4, 35.7, 24.8, 24.7],
    "8": [42.6, 21.0, 17.5, 17.4],
    "9": [51.1, 29.2, 21.2, 21.3],
    "10": [61.7, 36.3, 25.5, 25.3],
}

bloodfed_female_tsetsedf = pd.DataFrame(
    blood_female_tsetse, index=["0 h", "24 h", "72 h", "120 h"]
)
bloodfed_female_tsetsedf["time test"] = ["0 h", "24 h", "72 h", "120 h"]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 8))

plt.subplots_adjust(hspace=0.5)

ax.set_xticks([])
ax2.set_xticks([])

width_bar = 0.5

ticks = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
a = [x - 0.5 for x in ticks]
b = [x + 1 for x in ticks]

colors = plt.cm.YlOrBr(np.linspace(0.3, 0.7, 4))
colors2 = plt.cm.Reds(np.linspace(0.3, 0.7, 4))

ax.bar(
    a,
    unfed_female_tsetsedf.iloc[0].values.tolist()[:-1],
    width=width_bar,
    label="0 h",
    color=colors[::-1][0],
)

ax.bar(
    ticks,
    unfed_female_tsetsedf.iloc[1].values.tolist()[:-1],
    width=width_bar,
    label="24 h",
    color=colors[::-1][1],
)

ax.bar(
    [x + 0.5 for x in ticks],
    unfed_female_tsetsedf.iloc[2].values.tolist()[:-1],
    width=width_bar,
    label="72 h",
    color=colors[::-1][2],
)


ax.bar(
    [x + 1 for x in ticks],
    unfed_female_tsetsedf.iloc[3].values.tolist()[:-1],
    width=width_bar,
    label="120 h",
    color=colors[::-1][3],
)

table(
    ax,
    unfed_female_tsetsedf.drop(["time test"], axis=1),
    loc="bottom",
    rowColours=colors[::-1],
)


###########

ax2.bar(
    a,
    bloodfed_female_tsetsedf.iloc[0].values.tolist()[:-1],
    width=width_bar,
    label="0 h",
    color=colors2[::-1][0],
)

ax2.bar(
    ticks,
    bloodfed_female_tsetsedf.iloc[1].values.tolist()[:-1],
    width=width_bar,
    label="24 h",
    color=colors2[::-1][1],
)

ax2.bar(
    [x + 0.5 for x in ticks],
    bloodfed_female_tsetsedf.iloc[2].values.tolist()[:-1],
    width=width_bar,
    label="72 h",
    color=colors2[::-1][2],
)


ax2.bar(
    [x + 1 for x in ticks],
    bloodfed_female_tsetsedf.iloc[3].values.tolist()[:-1],
    width=width_bar,
    label="120 h",
    color=colors2[::-1][3],
)


table(
    ax2,
    unfed_female_tsetsedf.drop(["time test"], axis=1),
    loc="bottom",
    rowColours=colors2[::-1],
)


ax.set_title("Unfed female tsetse", fontweight="bold")
ax2.set_title("Bloodfed female tsetse", fontweight="bold")
ax.set_ylabel("weight (mg)")
ax2.set_ylabel("weight (mg)")

ax.grid(axis="y", alpha=0.3)
ax2.grid(axis="y", alpha=0.3)

plt.savefig("./results/plots/Fig2.tiff", dpi=350, bbox_inches='tight')
