import os
import glob
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import textdistance

def load_files(input_dir):
    """Load all text files from input directory."""
    file_paths = glob.glob(os.path.join(input_dir, "*.txt"))
    file_contents = {}
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            file_contents[filename] = content
    
    return file_contents

def calculate_cosine_similarity(file_contents, output_csv):
    """Calculate Cosine Similarity using TF-IDF."""
    file_names = list(file_contents.keys())
    n_files = len(file_names)
    print(f"Processing {n_files} files for Cosine similarity...")
    
    results = []
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
                    similarity = 1.0 if i == j else 0.0
                else:
                    similarity = dot_product / (norm1 * norm2)
                
                results.append({
                    'file1': file1, 
                    'file2': file2, 
                    'similarity': similarity
                })
        
        print(f"\nCompleted cosine similarity calculation in {time.time() - start_time:.2f} seconds")
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        return True
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return False

def calculate_edit_similarity(file_contents, output_csv):
    """Calculate Edit Distance Similarity."""
    file_names = list(file_contents.keys())
    n_files = len(file_names)
    print(f"Processing {n_files} files for Edit Distance similarity...")
    
    results = []
    start_time = time.time()
    
    for i, file1 in enumerate(file_names):
        print(f"Processing file {i+1}/{n_files}: {file1}", end='\r')
        for j in range(i, len(file_names)):
            file2 = file_names[j]
            
            if i == j:
                similarity = 1.0
            else:
                text1 = file_contents[file1][:5000]
                text2 = file_contents[file2][:5000]
                similarity = textdistance.levenshtein.normalized_similarity(text1, text2)
            
            results.append({
                'file1': file1, 
                'file2': file2, 
                'similarity': similarity
            })
    
    print(f"\nCompleted edit distance calculation in {time.time() - start_time:.2f} seconds")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return True

def calculate_jaccard_similarity(file_contents, output_csv):
    """Calculate Jaccard Similarity."""
    file_names = list(file_contents.keys())
    n_files = len(file_names)
    print(f"Processing {n_files} files for Jaccard similarity...")
    
    results = []
    start_time = time.time()
    
    for i, file1 in enumerate(file_names):
        print(f"Processing file {i+1}/{n_files}: {file1}", end='\r')
        for j in range(i, len(file_names)):
            file2 = file_names[j]
            
            if i == j:
                similarity = 1.0
            else:
                tokens1 = set(file_contents[file1].lower().split())
                tokens2 = set(file_contents[file2].lower().split())
                
                if not tokens1 or not tokens2:
                    similarity = 0.0
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
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return True

def calculate_all_similarities(input_dir, results_dir):
    """Calculate all similarity metrics."""
    # Load all files
    file_contents = load_files(input_dir)
    
    # Jaccard similarity
    print("Computing Jaccard similarity...")
    jaccard_csv = os.path.join(results_dir, "jaccard_similarity.csv")
    calculate_jaccard_similarity(file_contents, jaccard_csv)
    
    # Cosine similarity
    print("\nComputing Cosine similarity...")
    cosine_csv = os.path.join(results_dir, "cosine_similarity.csv")
    calculate_cosine_similarity(file_contents, cosine_csv)
    
    # Edit distance similarity
    print("\nComputing Edit Distance similarity...")
    edit_csv = os.path.join(results_dir, "edit_similarity.csv")
    calculate_edit_similarity(file_contents, edit_csv)
    
    return jaccard_csv, cosine_csv, edit_csv