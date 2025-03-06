# Haunted Places Similarity Analysis

## Project Description

This project analyzes a dataset of haunted places using multiple text similarity metrics to discover patterns and relationships between different haunted locations. It processes raw TSV data, converts it to structured JSON, breaks it into individual text files, then applies Jaccard, Cosine, and Edit Distance similarity metrics to quantify relationships between locations.

## Features
- End-to-end data processing pipeline
- Multiple text similarity metrics (Jaccard, Cosine, Edit Distance)
- Interactive visualizations showing similarity relationships
- Comprehensive similarity analysis report with statistical insights
- Links to original documents for further investigation

## Requirements
- Python 3.8+
- Required packages:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
textdistance
```
## Installation 
1. Clone the repository:

```
git clone https://github.com/yourusername/haunted_places_tika_analysis.git
cd haunted_places_tika_analysis
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Ensure your dataset is placed in the correct location:

```
haunted_places_tika_analysis/datasets/merged_dataset.tsv
```

## Usage
Running the Analysis Pipeline
To run the complete anlysis Pipeline:

```
python run_tika_similarity.py
```

This will:

1. Convert TSV data to JSON
2. Break JSON into individual text files
3. Calculate similarity metrics (Jaccard, Cosine, Edit Distance)
4. Generate a comprehensive analysis report with visualizations

Forcing Regeneration
To force regeneration of intermediate files:

```
python run_tika_similarity.py --force
```

## Project Structure
```
haunted_places_tika_analysis/
├── datasets/
│   └── merged_dataset.tsv         # Source data
├── haunted_places_files/          # Generated text files
├── similarity_results/            # Similarity metrics and report
│   ├── visualizations/            # Generated charts and graphs
│   └── similarity_analysis_report.md # Final analysis report
├── convert_tsv_to_json.py         # TSV to JSON converter
├── break_json.py                  # JSON to text files converter
├── similarity_calculator.py       # Similarity metrics implementation
├── report_generator.py            # Report generation module
└── run_tika_similarity.py         # Main script to run analysis
```
## Output and Results
The final analysis report is saved as a Markdown file in the `similarity_results` directory. It includes:

- Summary statistics for each similarity metric
- Visualizations showing relationships between locations
- Links to original documents for further investigation
