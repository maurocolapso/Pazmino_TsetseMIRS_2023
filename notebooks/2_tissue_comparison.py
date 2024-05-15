import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import matplotlib.pyplot as plt
import umap
import numpy as np

# Import data
tsetse_data = pd.read_csv("./data/raw/TseTse_finaldatasetclean.csv")

# rename columns and replace labels
tsetse_data.rename(
    columns={"Cat1": "Plate", "Cat3": "Sex", "Cat4": "Age", "Cat5": "Tissue"},
    inplace=True,
)

tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("T", "Thorax")
tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("H", "Head")
tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("A", "Abdomen")

# copy data and extract descriptos for slicing
tsetse_data_copy = tsetse_data.copy()
dList = ["Plate", "Sex", "Age", "ID.1", "Tissue", "ID"]
descriptorsDF = tsetse_data_copy[dList]

# Sorting females from each tissue
tsetse_females_thorax = tsetse_data_copy.loc[
    (descriptorsDF["Sex"] == "f") & (descriptorsDF["Tissue"] == "Thorax")
]
tsetse_females_head = tsetse_data_copy.loc[
    (descriptorsDF["Sex"] == "f") & (descriptorsDF["Tissue"] == "Head")
]
tsetse_females_abdomen = tsetse_data_copy.loc[
    (descriptorsDF["Sex"] == "f") & (descriptorsDF["Tissue"] == "Abdomen")
]

# make copy of the data
tsetse_females_thorax_copy = tsetse_females_thorax.copy()
tsetse_females_head_copy = tsetse_females_head.copy()
tsetse_females_abdomen_copy = tsetse_females_abdomen.copy()


# create descriptors and drop from the main data sets
dList = ["Plate", "Sex", "Age", "ID.1", "Tissue", "ID"]

descriptorsDF_thorax = tsetse_females_thorax[dList]
tsetse_females_thorax_copy.drop(dList, axis=1, inplace=True)

descriptorsDF_head = tsetse_females_head_copy[dList]
tsetse_females_head_copy.drop(dList, axis=1, inplace=True)

# extract wavenumbers to plot
waveNumslist = tsetse_females_thorax_copy.columns.values.tolist()
wavenumbers = [int(x) for x in waveNumslist]


y_labels_head = descriptorsDF_head["Age"]
y_labels_thorax = descriptorsDF_thorax["Age"]


# sorting males
tsetse_males_thorax = tsetse_data_copy.loc[
    (descriptorsDF["Sex"] == "m") & (descriptorsDF["Tissue"] == "Thorax")
]

tsetse_males_head = tsetse_data_copy.loc[
    (descriptorsDF["Sex"] == "m") & (descriptorsDF["Tissue"] == "Head")
]

tsetse_males_thorax_copy = tsetse_males_thorax.copy()
tsetse_males_head_copy = tsetse_males_head.copy()


colorpal = ["#999933", "#DDCC77", "#332288"]

# Import data
tsetse_data = pd.read_csv("./data/raw/TseTse_finaldatasetclean.csv")


# rename columns and replace labels
tsetse_data.rename(
    columns={"Cat1": "Plate", "Cat3": "Sex", "Cat4": "Age", "Cat5": "Tissue"},
    inplace=True,
)

tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("T", "Thorax")
tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("H", "Head")
tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("A", "Abdomen")


# copy data and sort by bodyparts
tsetse_data_copy = tsetse_data.copy()

tsetse_females_thorax = tsetse_data_copy.loc[
    (tsetse_data_copy["Sex"] == "f") & (tsetse_data_copy["Tissue"] == "Thorax")
]

tsetse_females_head = tsetse_data_copy.loc[
    (tsetse_data_copy["Sex"] == "f") & (tsetse_data_copy["Tissue"] == "Head")
]

tsetse_females_abdomen = tsetse_data_copy.loc[
    (tsetse_data_copy["Sex"] == "f") & (tsetse_data_copy["Tissue"] == "Abdomen")
]


tsetse_females_thorax_copy = tsetse_females_thorax.copy()
tsetse_females_head_copy = tsetse_females_head.copy()

# select the region 1800 - 900
tsetse_data_copy_onlywvns = tsetse_data_copy.loc[:, "1800":"900"]
tse_for_wvn = tsetse_females_head_copy.loc[:, "4000":"402"]

# create second list of wavenumbers
waveNumslist2 = tse_for_wvn.columns.values.tolist()
wavenumbers2 = [int(x) for x in waveNumslist2]


# UMAP
reducer = umap.UMAP(n_neighbors=200, transform_seed=124, min_dist=0.1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tsetse_data_copy_onlywvns)
embedding_head = reducer.fit_transform(features_scaled)

# PLOT

plt.rcParams["font.size"] = 12
sn.set_style("ticks")
sn.set_palette(sn.color_palette(colorpal))

colorpal = ["#999933", "#DDCC77", "#332288"]

fig = plt.figure(layout=None, figsize=(7.5, 4))
gs = fig.add_gridspec(nrows=3, ncols=2, hspace=0.4, wspace=0.6)

ax0 = fig.add_subplot(gs[1:3, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, 1])

for i in [ax1, ax2]:
    i.set_xlim(1800, 900)
    i.tick_params(left=False, right=False, labelleft=True, labelbottom=False)


ax3.set_xlabel("Wavenumbers (cm$^{-1}$)")
ax3.set_xlim(1800, 900)

sn.scatterplot(
    x=embedding_head[:, 0],
    y=embedding_head[:, 1],
    hue=tsetse_data_copy["Tissue"],
    ax=ax0,
    legend=True,
)
ax0.legend()

sn.despine(ax=ax0)
ax0.set_xlabel("UMAP1")
ax0.set_ylabel("UMAP2")

ax0.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)


# head
ax1.plot(
    wavenumbers2,
    np.mean(tsetse_females_head_copy.loc[:, "4000":"402"], axis=0),
    color="#DDCC77",
)

ax1.fill_between(
    wavenumbers2,
    y1=np.mean(tsetse_females_head_copy.loc[:, "4000":"402"], axis=0),
    y2=np.mean(tsetse_females_head_copy.loc[:, "4000":"402"], axis=0)
    + np.std(tsetse_females_head_copy.loc[:, "4000":"402"], axis=0),
    alpha=0.2,
    color="#DDCC77",
)

ax1.fill_between(
    wavenumbers2,
    y1=np.mean(tsetse_females_head_copy.loc[:, "4000":"402"], axis=0),
    y2=np.mean(tsetse_females_head_copy.loc[:, "4000":"402"], axis=0)
    - np.std(tsetse_females_head_copy.loc[:, "4000":"402"], axis=0),
    alpha=0.2,
    color="#DDCC77",
)


# thorax
ax2.plot(
    wavenumbers2,
    np.mean(tsetse_females_thorax_copy.loc[:, "4000":"402"], axis=0),
    color="#332288",
    label="thorax",
)

ax2.fill_between(
    wavenumbers2,
    y1=np.mean(tsetse_females_thorax_copy.loc[:, "4000":"402"], axis=0),
    y2=np.mean(tsetse_females_thorax_copy.loc[:, "4000":"402"], axis=0)
    + np.std(tsetse_females_thorax_copy.loc[:, "4000":"402"], axis=0),
    alpha=0.2,
    color="#332288",
)

ax2.fill_between(
    wavenumbers2,
    y1=np.mean(tsetse_females_thorax_copy.loc[:, "4000":"402"], axis=0),
    y2=np.mean(tsetse_females_thorax_copy.loc[:, "4000":"402"], axis=0)
    - np.std(tsetse_females_thorax_copy.loc[:, "4000":"402"], axis=0),
    alpha=0.2,
    color="#332288",
)


# abdomen
ax3.plot(
    wavenumbers2,
    np.mean(tsetse_females_abdomen_copy.loc[:, "4000":"402"], axis=0),
    color="#999933",
    label="Abdomen",
)

ax3.fill_between(
    wavenumbers2,
    y1=np.mean(tsetse_females_abdomen_copy.loc[:, "4000":"402"], axis=0),
    y2=np.mean(tsetse_females_abdomen_copy.loc[:, "4000":"402"], axis=0)
    + np.std(tsetse_females_abdomen_copy.loc[:, "4000":"402"], axis=0),
    alpha=0.2,
    color="#999933",
)

ax3.fill_between(
    wavenumbers2,
    y1=np.mean(tsetse_females_abdomen_copy.loc[:, "4000":"402"], axis=0),
    y2=np.mean(tsetse_females_abdomen_copy.loc[:, "4000":"402"], axis=0)
    - np.std(tsetse_females_abdomen_copy.loc[:, "4000":"402"], axis=0),
    alpha=0.2,
    color="#999933",
)

ax2.set_ylabel("Absorbance (a.u)")

labeles = ["A", "B", "C", "D", "E", "F"]
axes = fig.get_axes()
for a, l in zip(axes, labeles):
    a.set_title(l, y=1, loc="left", fontsize=12, fontweight="bold")

for i in [ax1, ax2, ax3]:
    i.set_ylim(0, 0.4)

sn.move_legend(ax0, "upper right", bbox_to_anchor=(1.25, 1.45))


fig.savefig("./results/plots/Fig2.png", dpi=300, bbox_inches="tight")
