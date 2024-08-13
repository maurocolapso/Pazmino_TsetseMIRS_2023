"""Preprocess the data"""

import pandas as pd

# import data
tsetse_data = pd.read_csv("./data/raw/TseTse_finaldatasetclean.csv", sep=",")

# rename columns
tsetse_data.rename(
    columns={"Cat1": "Plate", "Cat3": "Sex", "Cat4": "Age", "Cat5": "Tissue"},
    inplace=True,
)
tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("T", "Thorax")
tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("H", "Head")
tsetse_data["Tissue"] = tsetse_data["Tissue"].str.replace("A", "Abdomen")

# export final dataset

tsetse_data.to_csv("./data/processed/TseTse_processed.csv", index=False)
