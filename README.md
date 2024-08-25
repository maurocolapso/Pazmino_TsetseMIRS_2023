# Advancing age grading techniques for Glossina morsitans morsitans, vectors of African trypanosomiasis, through mid-infrared spectroscopy and machine learning

Manuscript code

## Data
- Individual raw files and assembled file can be downloaded from the Enlighten reporistory at: https://researchdata.gla.ac.uk/1564/
- Files were assembled using [BadBlood 1.1](https://github.com/magonji/MIMI-project).

## Scripts/Notebooks

- [Data preparation](/notebooks/1_data_preparation.py)
- [Tissue comparison](/notebooks/2_tissue_comparison.py)
- [Clustering](/notebooks/3_clustering.py)
- [Permutation test](/notebooks/4_Permutation_test.ipynb)
- [Bias test](/notebooks/5_Bias_testing.ipynb)
- [Sex prediction with all wavenumbers](/notebooks/6_Prediction_wholerange.ipynb)
- [Sex and age prediction using informative region](/notebooks/7_Prediction_restricted.ipynb)
- [Sample dessication times (figure)](/notebooks/Dessicationtimes.py)
- [Spectra comparison (figure)](/notebooks/spectra_comparison.py)



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

## Reproducing the analysis 

1. Download individual raw files
2. Create the assembled file using BadBlood 1.1 script. 
3. Create the final assembled file called 'TseTse_processed.csv" using [1_data_preparation.py](/notebooks/1_data_preparation.py) script.

