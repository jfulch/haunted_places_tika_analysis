#!/usr/bin/env python

import os
import sys
from similarity_calculator import calculate_all_similarities
from report_generator import generate_markdown_report
from convert_tsv_to_json import convert_tsv_to_json
from break_json import break_json

# Directories and files
input_tsv = "datasets/merged_dataset.tsv"  # Source TSV file
json_file = "datasets/merged_dataset.json"         # Intermediate JSON file
input_dir = "haunted_places_files"        # Directory to store individual text files
results_dir = "similarity_results"        # Directory to store similarity results

# Create output directories if they don't exist
for directory in [input_dir, results_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    print("Starting end-to-end haunted places analysis...")
    
    # Step 1: Convert TSV to JSON
    print("\nStep 1: Converting TSV to JSON...")
    force_regenerate = "--force" in sys.argv
    
    if not os.path.exists(json_file) or force_regenerate:
        convert_tsv_to_json(input_tsv, json_file)
    else:
        print(f"Using existing JSON file: {json_file} (use --force to regenerate)")
    
    # Step 2: Break JSON into individual text files
    print("\nStep 2: Breaking JSON into individual text files...")
    # Check if any text files already exist
    if os.path.exists(input_dir) and os.listdir(input_dir) and not force_regenerate:
        print(f"Text files directory {input_dir} already contains files.")
        print("Skipping text file generation (use --force to regenerate)")
    else:
        # Call the break_json function with the correct file path
        break_json(json_file, input_dir)
        print(f"Text files generated in {input_dir}/")
    
    # Step 3: Calculate all similarity metrics
    print("\nStep 3: Calculating similarity metrics...")
    jaccard_csv, cosine_csv, edit_csv = calculate_all_similarities(input_dir, results_dir)
    
    # Step 4: Generate report
    print("\nStep 4: Generating analysis report...")
    report_path = generate_markdown_report(jaccard_csv, cosine_csv, edit_csv)
    print(f"Report saved to: {report_path}")
    
    print("\nHaunted places analysis complete!")
    print("=" * 50)
    print(f"- Source TSV:  {input_tsv}")
    print(f"- JSON file:   {json_file}")
    print(f"- Text files:  {input_dir}/")
    print(f"- Results:     {results_dir}/")
    print(f"- Report:      {report_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()