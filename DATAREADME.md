# TITLE OF THE DATA SET: Raw and assembled mid-infrared spectroscopy data from laboratory reared samples of of Glossina spp.
 
## CONTACT INFORMATION
Author/Associate or Co-investigator Information

- Name: Mauro Pazmiño Betancourth
- ORCID: 0000-0003-4259-0611
- Institution: University of Glasgow
- Address:  Graham Kerr Building
- E-mail: mauro.pazminobetancourth@glasgow.ac.uk

Author/Alternate Contact Information
- Name: Ivan Casas
- ORCID: 0009-0004-8017-2079
- Institution: Univeristy of Glasgow
- Address: Graham Kerr Building
- Email: 2332990C@student.gla.ac.uk

Date of data collection: 2022-12-01 - 2023-03-31 

Geographic location of data collection: Samples measured at Joseph Black Building, University of Glasgow, UK.


## SHARING/ACCESS INFORMATION

Was data derived from another source? No

## DATA & FILE OVERVIEW

Filelist: 
```
Data
├── ATR_data
│   ├── PLATE1_ALL
│   ├── PLATE2_ALL
│   ├── PLATE3_ALL
│   ├── PLATE4_ALL
│   ├── PLATE5_ALL
│   ├── PLATE6_ALL
│   ├── PLATE7_ALL
│   └── PLATE8_ALL
├── Assembled_data
│   └── TseTse_finaldatasetclean.csv
└── README.rtf
```
Recommended citation for this dataset: 

## METHODOLOGICAL INFORMATION
An age-stratified colony of Glossina morsitans morsitans Westwood, established in 2004 at the Liverpool School of Tropical Medicine, UK, was daily maintained at 26 - 28°C, 68 – 78 % humidity with a 12 h/12 h light/dark cycle. The colony was fed three times a week on sterile defibrinated horse blood (TCS Biosciences Ltd, Buckingham, UK). 
Spectra from individual heads, thoraces and abdomens were taken by Attenuated Total Reflection FT-IR spectroscopy using a Bruker ALPHA II spectrometer equipped with a Globar lamp, a DLaTGS detector, a KBr beamsplitter, and a diamond ATR accessory (Bruker Platinum ATR Unit A225). Twenty-four scans were taken at room temperature between 4000 and 400 cm−1 with 4 cm−1 resolution. Data collected by Mauro Pazmino and Ivan Casas.

## FILE NAMING CONVENTION:
Raw files: PLATE1-01-f-3d-A-20230124.dpt

### Attributes: 
1. PLATE1: Plate where the sample was stored.
2. 01: Two-digit code assigned to the sample in the lab
3. f: Sex of the sample (m: male, f: female)
4. 3d: Chronological age of the sample in days
5. A: One letter digit that indicate which body part was measured (H: head, T: Thorax, A: Abodomen)
6. 20230124: Date at which the sample was measured in YYYYMMDD format


File format: OPUS format and data point table (.dpt)

Example: PLATE1-01-f-3d-A-20230124.dpt


## DATA-SPECIFIC INFORMATION FOR: 
TseTse_finaldatasetclean.csv

Final data set assembled using Bad Blood script (https://github.com/magonji/MIMI-project/blob/master/Bad%20Blood%201.1.ipynb)

Number of variables: 5 categorical variables and 1800 numerical variables

Number of cases/rows: 1188 rows

Variable List: 
Plate, ID sample, sex, age, body part, date measured, and wavenumbers from 4000 - 401 cm-1
