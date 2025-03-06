# Haunted Places Tika Analysis

This repository contains the code to analyze the text of haunted places using the Tika library.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the analysis you first need to manually upload your tsv file to the `dataset` folder. The code is currently loking for a hardcoded value called `merged_dataset.tsv`. Once you have uploaded your file, you can run the following command:

```bash
python run_tika_similarity.py
```

The output of this script will:

1. Extract the text from the tsv file
2. Convert the tsv to a single json object called `merged_dataset.json` in the datasets folder
3. Break the single json file into smaller json files of 1000 records each in the datasets folder
4. Run the Tika analysis on each of the smaller json files and output a csv file for each similarity metric in the datasets folder
5. Run anlysis to generate visualizations for the similarity metrics in the datasets folder
6. Generate a markdown report (here)[/similarity_results/similarity_analysis_report.md]
