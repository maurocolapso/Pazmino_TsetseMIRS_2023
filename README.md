# Mid-infrared spectroscopy and machine learning for vector surveillance in tsetse (Glossina spp.)

Manuscript code

## Data
- Individual raw files can be downloaded from elighten reporistory at:
- Files were assembled using BadBlood from

## Scripts/Notebooks

- [Data preparation](/notebooks/1_data_preparation.py)
- [Tissue comparison](/notebooks/2_tissue_comparison.py)
- [Clustering](/notebooks/3_clustering.py)
- [Permutation test](/notebooks/4_Permutation_test.ipynb)
- [Bias test](/notebooks/5_Bias_testing.ipynb)
- [Sex and age prediction using informative region](/notebooks/7_Prediction_restricted.ipynb)
- [Sex prediction with all wavenumbers](/notebooks/6_Prediction_wholerange.ipynb)



## File structure

```
tsetse_MIRS
├── README.md
├── data
│   ├── processed
│   └── raw
├── notebooks
│   ├── 1_data_preparation.py
│   ├── 2_tissue_comparison.py
│   ├── 3_clustering.py
│   ├── 4_Permutation_test.ipynb
│   ├── 5_Bias_testing.ipynb
│   ├── 6_Prediction_wholerange.ipynb
│   ├── 7_Prediction_restricted.ipynb
├── results
│   ├── plots
│   └── tables
└── src
    └── utilities.py
```
