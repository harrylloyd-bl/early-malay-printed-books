# Early Malay Printed Books Metadata Extraction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Extracting metadata from the Early Malay Printed Books catalogue created by Ian Proudfoot. As part of the Bollinger Early Malay Printing Digitisation project 250 works from the BL's early Malay printed are being digitised. This repo contains the code to generate RDA metadata for those works by extracting the OCR text from Proudfoot into individual catalogue records, then using Qwen to structure that text into RDA fields.

3 step process  
1. Pre-process the OCR from the printed catalogue into text for individual catalogue entries that can be mapped to the works 
2. Pass to Qwen to structure into RDA
3. Post-processing to tidy up the extracted fields 

## Project Organization

```
├── LICENSE            <- MIT License
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         emp and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── uv.lock   <- The uv lock file for reproducing the analysis environment
│
└── emp   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes emp a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

