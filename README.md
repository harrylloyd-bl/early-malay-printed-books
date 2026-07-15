# Early Malay Printed Books Metadata Extraction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Extracting metadata from the Early Malay Printed Books catalogue created by Ian Proudfoot. As part of the Bollinger Early Malay Printing Digitisation project 251 works from the BL's early Malay printed are being digitised. This repo contains the code to generate MARC/RDA metadata for those works by extracting the OCR text from Proudfoot into individual catalogue records, then using Qwen to structure that text into RDA fields. The BL Asian and African collection contains another 433 works that are listed in Proudfoot. As I'd developed the infrastructure to process the Bollinger set we decided to also process the remaining 433, this also supported the replacement of some of the original 251 that were removed from digitisation due to conservation concerns.

4 step process  
1. Pre-process the OCR from the printed catalogue into text for individual catalogue entries that are mapped to the works. Verify that all shelfmarks in the list to be extracted are present
2. Pass to Qwen to structure into JSON with fields that map to MARC
3. Post-process to map JSON to MARC and clean extracted fields 
4. Verify all Bollinger/non-Bollinger shelfmarks have been extracted and export to csv/xlsx

## Project Organization

```
├── LICENSE            <- MIT License
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources including Proudfoot AAC lists.
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
├── uv.lock   <- The uv lock file for reproducing the analysis environment
│
└── emp   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes emp a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
```


## How to run
This project is package managed using [UV](https://docs.astral.sh/uv/getting-started/), so will automatically install packages from the uv.lock file when running commands. If using a different package manager the pyproject.toml file has a human readable list of packages.

The bulk of the time will be spent verifying shelfmark coverage and in post-processing to make sure that all shelfmarks have entries. This is due to poor OCR in Proudfoot requiring a lot of manual mapping between titles/shelfmarks extracted from Proudfoot and whatever list of titles you are trying to generate metadata for.

### Accessing data
Get a copy of Proudfoot with OCR from https://mcp.anu.edu.au/proudfoot/EMPB/web/pdf/Complete.pdf and save it as 'emp.pdf' in data/raw.
Get a copy of Proudfoot-BL collection-6.10.25.xlsx from Annabel Gallop and extract CSVs for the 251 Bollinger titles and the 684 Full AAC list, which will be processed into the 433 non Bollinger AAC titles.

### Creating title_loc_df
Run main.py with the `CREATE_TITLE_LOC_DF = True` in main() to create title_loc_df, which contains entry titles and entry text for all catalogue entries in Proudfoot. Behind the scenes this processes emp.pdf in data/raw to extract a list of titles then uses that to extract all catalogue entries with entry text.

### Verify shelfmark coverage
Use 02-HL-prepare_entries.ipynb to check that all Shelfmarks in the list of titles you want to extract are contained in title_loc_df. At this point you will have to do a lot of manual mapping due to the very variable quality of OCR for this copy of Proudfoot. Expect maybe 30% of titles to need some kind of automated or manual correction to map correctly. Export a clean deduplicated list of entries to data/interim and copy the filepath to `PREPARED_ENTRIES` variable in main().

### Pass entries to Qwen model
Get an API key for Qwen services here: https://modelstudio.console.alibabacloud.com. Store it in a .env file and follow the Qwen documentation to get any other secrets needed for the .env file. At the time of writing this included an API host, which is your Workspace ID in the Model Studio console joined to a web address for servers in a specific location.

Set `BATCH` with a sensible batch name - I used date
Select `MODEL` if upgrading from qwen3.7
Set `USE_DEFAULT_WORKSPACE = False` if not using the default Singapore server location. 
Set `POST_PROCESS = True`
Run main.py with `CALL_API = True` in main()

Output will be written to a folder called `{BATCH}` in data/processed

### Post-process  
Use 04-HL-select_editions.ipynb to verify that titles and editions corresponding to all shelfmarks in the required list have been extracted. If any haven't been extracted verify why, normally due to poor OCR requiring manual setting of a shelfmark to an edition in the output. 
Once verified that all shelfmarks have been extracted export as csv/xlsx

--------

