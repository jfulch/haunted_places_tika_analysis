#!/usr/bin/env python
import os
import sys
import subprocess
import glob
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import textdistance
import datetime

# Set the path to the actual tika-similarity directory
TIKA_SIMILARITY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tika-similarity")

print(f"Using tika-similarity path: {TIKA_SIMILARITY_PATH}")
print("Starting similarity analysis...")

# Directories
input_dir = "haunted_places_files"  # Directory with your text files
results_dir = "similarity_results"   # Directory to store results

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Use the same Python interpreter that's running this script
# This ensures we use the correct Python executable
PYTHON_EXECUTABLE = sys.executable
print(f"Using Python executable: {PYTHON_EXECUTABLE}")

# Search for scripts recursively
def find_script(base_dir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            if pattern in filename.lower() and filename.endswith('.py'):
                matches.append(os.path.join(root, filename))
    return matches

# Print the directory structure for debugging
def print_directory_structure(base_dir, level=0):
    print('  ' * level + os.path.basename(base_dir) + '/')
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                print_directory_structure(item_path, level+1)
            elif item.endswith('.py'):
                print('  ' * (level+1) + item)

# Print directory structure for debugging
print("\nDirectory structure of tika-similarity:")
print_directory_structure(TIKA_SIMILARITY_PATH)
print("\n")

# Run a specific similarity script
def run_similarity_script(script_type, output_csv):
    # Search for scripts matching the type (jaccard, cosine, edit)
    potential_scripts = find_script(TIKA_SIMILARITY_PATH, script_type)
    
    if potential_scripts:
        script_path = potential_scripts[0]  # Use the first match
        print(f"Found script: {script_path}")
        
        try:
            print(f"Running: {PYTHON_EXECUTABLE} {script_path} --inputDir {input_dir} --outCSV {output_csv}")
            subprocess.run([
                PYTHON_EXECUTABLE,  # Use the correct Python executable
                script_path, 
                "--inputDir", input_dir, 
                "--outCSV", output_csv
            ], check=True)
            print(f"Results saved to {output_csv}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_path}: {e}")
            return False
    else:
        print(f"Error: Could not find any {script_type} similarity script in {TIKA_SIMILARITY_PATH}")
        return False

# Custom similarity calculation function
def calculate_custom_similarity(similarity_type, output_csv):
    """Calculate similarity using custom implementation."""
    print(f"Using custom implementation for {similarity_type} similarity...")
    
    # Load all text files from input directory
    file_paths = glob.glob(os.path.join(input_dir, "*.txt"))
    file_contents = {}
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            file_contents[filename] = content
    
    file_names = list(file_contents.keys())
    n_files = len(file_names)
    print(f"Processing {n_files} files...")
    
    results = []
    
    if similarity_type.lower() == "cosine":
        # Calculate Cosine Similarity using TF-IDF
        start_time = time.time()
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([file_contents[name] for name in file_names])
            
            for i, file1 in enumerate(file_names):
                print(f"Processing file {i+1}/{n_files}: {file1}", end='\r')
                for j in range(i, len(file_names)):
                    file2 = file_names[j]
                    
                    # Get TF-IDF vectors
                    vec1 = tfidf_matrix[i].toarray().flatten()
                    vec2 = tfidf_matrix[j].toarray().flatten()
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 == 0 or norm2 == 0:
                        similarity = 1.0 if i == j else 0.0  # Handle zero vectors
                    else:
                        similarity = dot_product / (norm1 * norm2)
                    
                    results.append({
                        'file1': file1, 
                        'file2': file2, 
                        'similarity': similarity
                    })
            
            print(f"\nCompleted cosine similarity calculation in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return False
    
    elif similarity_type.lower() == "edit":
        # Calculate Edit Distance Similarity
        start_time = time.time()
        for i, file1 in enumerate(file_names):
            print(f"Processing file {i+1}/{n_files}: {file1}", end='\r')
            for j in range(i, len(file_names)):
                file2 = file_names[j]
                
                # For identical files, set similarity to 1.0 without calculation
                if i == j:
                    similarity = 1.0
                else:
                    # Use normalized Levenshtein distance
                    # For efficiency with long texts, compare just the first 5000 characters
                    text1 = file_contents[file1][:5000]
                    text2 = file_contents[file2][:5000]
                    similarity = textdistance.levenshtein.normalized_similarity(text1, text2)
                
                results.append({
                    'file1': file1, 
                    'file2': file2, 
                    'similarity': similarity
                })
        
        print(f"\nCompleted edit distance calculation in {time.time() - start_time:.2f} seconds")
    
    elif similarity_type.lower() == "jaccard":
        # Calculate Jaccard Similarity
        start_time = time.time()
        for i, file1 in enumerate(file_names):
            print(f"Processing file {i+1}/{n_files}: {file1}", end='\r')
            for j in range(i, len(file_names)):
                file2 = file_names[j]
                
                # For identical files, set similarity to 1.0 without calculation
                if i == j:
                    similarity = 1.0
                else:
                    # Calculate Jaccard similarity using token sets
                    # Convert text to lowercase and split into words
                    tokens1 = set(file_contents[file1].lower().split())
                    tokens2 = set(file_contents[file2].lower().split())
                    
                    # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
                    if not tokens1 or not tokens2:
                        similarity = 0.0  # Handle empty sets
                    else:
                        intersection = len(tokens1.intersection(tokens2))
                        union = len(tokens1.union(tokens2))
                        similarity = intersection / union
                
                results.append({
                    'file1': file1, 
                    'file2': file2, 
                    'similarity': similarity
                })
        
        print(f"\nCompleted Jaccard similarity calculation in {time.time() - start_time:.2f} seconds")
    
    else:
        print(f"Unknown similarity type: {similarity_type}")
        return False
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return True

# Skip Tika attempts and use custom implementations for all similarity measures
print("Computing Jaccard similarity...")
jaccard_csv = os.path.join(results_dir, "jaccard_similarity.csv")
calculate_custom_similarity("jaccard", jaccard_csv)

print("\nComputing Cosine similarity...")
cosine_csv = os.path.join(results_dir, "cosine_similarity.csv")
calculate_custom_similarity("cosine", cosine_csv)

print("\nComputing Edit Distance similarity...")
edit_csv = os.path.join(results_dir, "edit_similarity.csv")
calculate_custom_similarity("edit", edit_csv)

print("\nSimilarity analysis complete!")