# import packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import matplotlib.pyplot as plt
import umap


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

dList = ["Plate", "Sex", "Age", "ID.1", "Tissue", "ID"]
descriptorsDF_thorax_males = tsetse_males_thorax[dList]
tsetse_males_thorax_copy.drop(dList, axis=1, inplace=True)


descriptorsDF_head_males = tsetse_males_head_copy[dList]
tsetse_males_head_copy.drop(dList, axis=1, inplace=True)


# UMAP
# females head
reducer = umap.UMAP(n_neighbors=30, transform_seed=123, min_dist=0.0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tsetse_females_head_copy)
embedding_head = reducer.fit_transform(features_scaled)

# Females thorax
reducer = umap.UMAP(n_neighbors=30, transform_seed=123, min_dist=0.0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tsetse_females_thorax_copy)
embedding_thorax = reducer.fit_transform(features_scaled)


# males head
reducer = umap.UMAP(n_neighbors=30, transform_seed=123, min_dist=0.0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tsetse_males_head_copy)
embedding_head_male = reducer.fit_transform(features_scaled)

# males thorax
reducer = umap.UMAP(n_neighbors=30, transform_seed=123, min_dist=0.0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tsetse_males_thorax_copy)
embedding_thorax_male = reducer.fit_transform(features_scaled)


# remove 3 days old for proper clustering between males and females
tsetse_data_sameages = tsetse_data[(tsetse_data["Age"] != "3d")]
tsetse_data_sameages_copy = tsetse_data_sameages.copy()

dList = ["Plate", "Sex", "Age", "ID.1", "Tissue", "ID"]
descriptorsDF_sameage = tsetse_data_sameages_copy[dList]
tsetse_data_sameages_copy.drop(dList, axis=1, inplace=True)

tsetse_head = tsetse_data_sameages_copy[(descriptorsDF_sameage["Tissue"] == "Head")]
tsetse_thorax = tsetse_data_sameages_copy[(descriptorsDF_sameage["Tissue"] == "Thorax")]


# UMAP
# head both sexes
reducer = umap.UMAP(n_neighbors=30, transform_seed=123, min_dist=0.0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tsetse_head)
embedding_head_sex = reducer.fit_transform(features_scaled)

# thorax both sexes
reducer = umap.UMAP(n_neighbors=30, transform_seed=123, min_dist=0.0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tsetse_thorax)
embedding_thorax_sex = reducer.fit_transform(features_scaled)


# Plot
colorpal = ["#332288", "#117733", "#44AA99"]
colorpal2 = ["#117733", "#44AA99"]

colorsex = ["#005AB5", "#DC3220"]
markpal = ["X", "o", "s"]
sn.set_palette(sn.color_palette(colorpal))

fig, ((ax, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 6))

plt.subplots_adjust(hspace=0.8)

# MALES AGE

sn.scatterplot(
    x=embedding_head_male[:, 0],
    y=embedding_head_male[:, 1],
    hue=descriptorsDF_head_males["Age"],
    hue_order=["5w", "7w"],
    alpha=0.8,
    palette=colorpal2,
    legend=True,
    ax=ax2,
)

sn.scatterplot(
    x=embedding_thorax_male[:, 0],
    y=embedding_thorax_male[:, 1],
    hue=descriptorsDF_thorax_males["Age"],
    hue_order=["5w", "7w"],
    palette=colorpal2,
    alpha=0.8,
    legend=True,
    ax=ax5,
)

# FEMALES AGE
# head
sn.scatterplot(
    x=embedding_head[:, 0],
    y=embedding_head[:, 1],
    alpha=0.8,
    hue=descriptorsDF_head["Age"],
    ax=ax3,
    legend=True,
)


# thorax
sn.scatterplot(
    x=embedding_thorax[:, 0],
    y=embedding_thorax[:, 1],
    alpha=1,
    hue=descriptorsDF_thorax["Age"],
    markers=markpal,
    legend=True,
    ax=ax6,
)


# SEX

sn.scatterplot(
    x=embedding_head_sex[:, 0],
    y=embedding_head_sex[:, 1],
    hue=descriptorsDF_sameage[descriptorsDF_sameage["Tissue"] == "Head"]["Sex"],
    palette=colorsex,
    alpha=0.8,
    ax=ax,
)


sn.scatterplot(
    x=embedding_thorax_sex[:, 0],
    y=embedding_thorax_sex[:, 1],
    hue=descriptorsDF_sameage[descriptorsDF_sameage["Tissue"] == "Thorax"]["Sex"],
    alpha=0.8,
    palette=colorsex,
    legend=True,
    ax=ax4,
)


ax.legend(
    ncol=2, bbox_to_anchor=(0.05, 1.1, 1, 0.1), loc="center", frameon=False, title="Sex"
)
ax2.legend(
    ncol=2,
    bbox_to_anchor=(0.05, 1.1, 1, 0.1),
    loc="center",
    frameon=False,
    title="Male age",
)
ax3.legend(
    ncol=3,
    bbox_to_anchor=(0.10, 1.1, 1, 0.1),
    loc="center",
    frameon=False,
    title="Female age",
)

ax4.legend(
    ncol=2, bbox_to_anchor=(0.05, 1.1, 1, 0.1), loc="center", frameon=False, title="Sex"
)
ax5.legend(
    ncol=2,
    bbox_to_anchor=(0.05, 1.1, 1, 0.1),
    loc="center",
    frameon=False,
    title="Male age",
)
ax6.legend(
    ncol=3,
    bbox_to_anchor=(0.10, 1.1, 1, 0.1),
    loc="center",
    frameon=False,
    title="Female age",
)

fig.text(0.1, 1, s="Head", fontsize=15, fontweight="bold")
fig.text(0.1, 0.5, s="Thorax", fontsize=15, fontweight="bold")

ax.set_ylabel("UMAP2")
ax4.set_ylabel("UMAP2")

ax4.set_xlabel("UMAP1")
ax5.set_xlabel("UMAP1")
ax6.set_xlabel("UMAP1")

ax.set_xlabel("UMAP1")
ax2.set_xlabel("UMAP1")
ax3.set_xlabel("UMAP1")

# ax3.legend( bbox_to_anchor=(0.5, 0.05, 0.3, 0.1),ncols=3,frameon=False)

labeles = ["A", "B", "C", "D", "E", "F"]
axes = fig.get_axes()
for a, l in zip(axes, labeles):
    a.set_title(l, loc="left", fontsize=20, fontweight="bold")

plt.savefig("./results/plots/Fig3.tiff", dpi=300)
