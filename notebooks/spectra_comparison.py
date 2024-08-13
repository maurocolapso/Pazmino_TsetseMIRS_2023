"""Recreate figure 5 of the manuscript"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

tsetse_data = pd.read_csv("./data/processed/TseTse_processed_wo_outliers.csv")
tsetse_females_thorax = tsetse_data.loc[
    (tsetse_data["Sex"] == "f") & (tsetse_data["Tissue"] == "Thorax")
]

tsetse_females_head = tsetse_data.loc[
    (tsetse_data["Sex"] == "f") & (tsetse_data["Tissue"] == "Head")
]

X_females_thorax = tsetse_females_thorax.loc[:, "4000":"402"]
X_females_head = tsetse_females_head.loc[:, "4000":"402"]

y_females_thorax = tsetse_females_thorax.loc[:, "Age"]
y_females_head = tsetse_females_head.loc[:, "Age"]

X_females_thorax = X_females_thorax[X_females_thorax.columns[::4]]
X_females_head = X_females_head[X_females_head.columns[::4]]

tsetse_males_thorax = tsetse_data.loc[
    (tsetse_data["Sex"] == "m") & (tsetse_data["Tissue"] == "Thorax")
]

tsetse_males_head = tsetse_data.loc[
    (tsetse_data["Sex"] == "m") & (tsetse_data["Tissue"] == "Head")
]

X_males_thorax = tsetse_males_thorax.loc[:, "4000":"402"]
X_males_head = tsetse_males_head.loc[:, "4000":"402"]

y_males_thorax = tsetse_males_thorax.loc[:, "Age"]
y_males_head = tsetse_males_head.loc[:, "Age"]

X_males_thorax = X_males_thorax[X_males_thorax.columns[::4]]
X_males_head = X_males_head[X_males_head.columns[::4]]

waveNumslist = X_females_thorax.columns.values.tolist()
wavenumbers = [int(x) for x in waveNumslist]

colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
colors2 = ["#d95f02", "#7570b3", "#e7298a", "#66a61e"]


rc = {
    "font.size": 12,
    "font.family": "Arial",
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

sn.set_style("ticks")
sn.set_context("notebook", rc=rc)

fig = plt.figure(layout=None, figsize=(10, 5))
gs = fig.add_gridspec(nrows=2,
                      ncols=2,
                      left=0.05,
                      right=0.75,
                      hspace=0.7,
                      wspace=0.3)


ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])


# females

for i, c in zip(np.unique(y_females_head), colors):
    sn.lineplot(
        x=wavenumbers,
        y=np.mean(X_females_head[y_females_head == i], axis=0),
        label=i,
        color=c,
        linewidth=1,
        ax=ax1,
    )

for i, c in zip(np.unique(y_females_thorax), colors):
    sn.lineplot(
        x=wavenumbers,
        y=np.mean(X_females_thorax[y_females_thorax == i], axis=0),
        label=i,
        color=c,
        linewidth=1,
        ax=ax2,
    )

# males

for i, c in zip(np.unique(y_males_head), colors2):
    sn.lineplot(
        x=wavenumbers,
        y=np.mean(X_males_head[y_males_head == i], axis=0),
        label=i,
        color=c,
        linewidth=1,
        ax=ax3,
    )



for i, c in zip(np.unique(y_males_thorax), colors2):
    sn.lineplot(
        x=wavenumbers,
        y=np.mean(X_males_thorax[y_males_thorax == i], axis=0),
        label=i,
        color=c,
        linewidth=1,
        ax=ax4,
    )


ax1.set_title("A", fontsize=16, loc="left", fontweight="bold")
ax2.set_title("B", fontsize=16, loc="left", fontweight="bold")
ax3.set_title("C", fontsize=16, loc="left", fontweight="bold")
ax4.set_title("D", fontsize=16, loc="left", fontweight="bold")
ax3.set_xlabel("Wavenumber (cm$^{-1}$)")
ax4.set_xlabel("Wavenumber (cm$^{-1}$)")

ax1.set_xlim(4000, 401)
ax2.set_xlim(4000, 401)
ax3.set_xlim(4000, 401)
ax4.set_xlim(4000, 401)

ax1.set_ylim(0, 0.35)
ax2.set_ylim(0, 0.35)
ax3.set_ylim(0, 0.35)
ax4.set_ylim(0, 0.35)

fig.text(x=-0.02,
         y=0.5, s="Absorbance (a.u)",
         rotation=90,
         verticalalignment="center")

fig.text(x=0.4, y=0.95, s="Females", ha="center", fontweight="bold")
fig.text(x=0.4, y=0.47, s="Males", ha="center", fontweight="bold")


plt.savefig("./results/plots/Fig5.tiff", dpi=300, bbox_inches="tight")
