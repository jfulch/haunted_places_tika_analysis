#!/usr/bin/env python
# filepath: /Users/jfulch/git/school/haunted_places_tika_analysis/run_tika_similarity.py
import os
import sys
from similarity_calculator import calculate_all_similarities
from report_generator import generate_markdown_report

# Directories
input_dir = "haunted_places_files"  # Directory with your text files
results_dir = "similarity_results"   # Directory to store results

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def main():
    print("Starting similarity analysis...")
    
    # Calculate all similarity metrics
    jaccard_csv, cosine_csv, edit_csv = calculate_all_similarities(input_dir, results_dir)
    
    # Generate report
    print("\nGenerating analysis report...")
    report_path = generate_markdown_report(jaccard_csv, cosine_csv, edit_csv)
    print(f"Report saved to: {report_path}")
    
    print("\nSimilarity analysis complete!")

if __name__ == "__main__":
    main()