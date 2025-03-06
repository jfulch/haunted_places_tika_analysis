import os
import glob
import time
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import textdistance
import multiprocessing
from joblib import Parallel, delayed
import tqdm

def load_files(input_dir, max_files=0):
    """Load all text files from input directory.
    
    Args:
        input_dir: Directory containing text files
        max_files: If > 0, limit to this many files (0 = all files)
    """
    file_paths = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if max_files > 0 and max_files < len(file_paths):
        print(f"Limiting to {max_files} files (from {len(file_paths)} total)")
        random.seed(42)  # For reproducibility
        file_paths = random.sample(file_paths, max_files)
    
    file_contents = {}
    print(f"Loading {len(file_paths)} files...")
    for file_path in tqdm.tqdm(file_paths, desc="Loading files"):
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            file_contents[filename] = content
    
    return file_contents

def _calculate_jaccard_for_pair(pair_data):
    """Calculate Jaccard similarity for a single pair of documents."""
    file_contents, file1, file2, i, j = pair_data
    
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
    
    return {'file1': file1, 'file2': file2, 'similarity': similarity}

def calculate_jaccard_similarity(file_contents, output_csv, n_jobs, sample_ratio=1.0):
    """Calculate Jaccard Similarity with parallel processing and sampling.
    
    Args:
        file_contents: Dictionary mapping filenames to file contents
        output_csv: Path to save results
        n_jobs: Number of CPU cores to use
        sample_ratio: Ratio of document pairs to sample (0.0-1.0)
    """
    file_names = list(file_contents.keys())
    n_files = len(file_names)
    print(f"Processing {n_files} files for Jaccard similarity using {n_jobs} CPU cores...")
    
    # Generate all possible pairs
    all_pairs = []
    for i, file1 in enumerate(file_names):
        for j in range(i, len(file_names)):
            file2 = file_names[j]
            all_pairs.append((i, j, file1, file2))
    
    # Sample pairs if requested
    total_pairs = len(all_pairs)
    sample_size = max(int(total_pairs * sample_ratio), 1)  # At least 1 pair
    
    # Always include all self-comparisons (i==j) in the sample
    self_pairs = [(i, i, file_names[i], file_names[i]) for i in range(n_files)]
    
    if sample_ratio < 1.0 and sample_size < total_pairs:
        # Set random seed for reproducibility
        random.seed(42)
        
        # Get non-self pairs for sampling
        non_self_pairs = [p for p in all_pairs if p[0] != p[1]]
        
        # Calculate how many non-self pairs to sample
        non_self_sample_size = min(sample_size - len(self_pairs), len(non_self_pairs))
        
        # Sample non-self pairs
        sampled_non_self = random.sample(non_self_pairs, non_self_sample_size)
        
        # Combine with self pairs
        sampled_pairs = self_pairs + sampled_non_self
        
        print(f"Sampling {len(sampled_pairs)} document pairs ({sample_ratio:.1%} of {total_pairs})")
        print(f"  - {len(self_pairs)} self comparisons (always included)")
        print(f"  - {len(sampled_non_self)} sampled non-self comparisons")
    else:
        sampled_pairs = all_pairs
        print(f"Calculating all {total_pairs} document pairs")
    
    # Convert sampled pairs to format needed for calculation
    pairs = [(file_contents, file1, file2, i, j) for i, j, file1, file2 in sampled_pairs]
    
    start_time = time.time()
    
    # Process in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_jaccard_for_pair)(pair_data) 
        for pair_data in tqdm.tqdm(pairs, desc="Jaccard Similarity")
    )
    
    print(f"Completed Jaccard similarity calculation in {time.time() - start_time:.2f} seconds")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return True

def _calculate_cosine_for_pair(pair_data):
    """Calculate Cosine similarity for a single pair using precomputed TF-IDF vectors."""
    tfidf_matrix, file_names, i, j = pair_data
    
    file1 = file_names[i]
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
    
    return {'file1': file1, 'file2': file2, 'similarity': similarity}

def calculate_cosine_similarity(file_contents, output_csv, n_jobs, sample_ratio=1.0):
    """Calculate Cosine Similarity using TF-IDF with parallelization and sampling."""
    file_names = list(file_contents.keys())
    n_files = len(file_names)
    print(f"Processing {n_files} files for Cosine similarity using {n_jobs} CPU cores...")
    
    start_time = time.time()
    
    try:
        # First vectorize all documents (this can't be parallelized easily)
        print("Generating TF-IDF matrix...")
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for speed
        tfidf_matrix = vectorizer.fit_transform([file_contents[name] for name in file_names])
        
        # Generate all possible pairs
        all_pairs = []
        for i in range(n_files):
            for j in range(i, n_files):
                all_pairs.append((i, j))
        
        # Sample pairs if requested
        total_pairs = len(all_pairs)
        sample_size = max(int(total_pairs * sample_ratio), 1)  # At least 1 pair
        
        # Always include all self-comparisons (i==j) in the sample
        self_pairs = [(i, i) for i in range(n_files)]
        
        if sample_ratio < 1.0 and sample_size < total_pairs:
            # Set random seed for reproducibility
            random.seed(42)
            
            # Get non-self pairs for sampling
            non_self_pairs = [p for p in all_pairs if p[0] != p[1]]
            
            # Calculate how many non-self pairs to sample
            non_self_sample_size = min(sample_size - len(self_pairs), len(non_self_pairs))
            
            # Sample non-self pairs
            sampled_non_self = random.sample(non_self_pairs, non_self_sample_size)
            
            # Combine with self pairs
            sampled_pairs = self_pairs + sampled_non_self
            
            print(f"Sampling {len(sampled_pairs)} document pairs ({sample_ratio:.1%} of {total_pairs})")
            print(f"  - {len(self_pairs)} self comparisons (always included)")
            print(f"  - {len(sampled_non_self)} sampled non-self comparisons")
        else:
            sampled_pairs = all_pairs
            print(f"Calculating all {total_pairs} document pairs")
        
        # Convert to format needed for parallel processing
        pairs = [(tfidf_matrix, file_names, i, j) for i, j in sampled_pairs]
        
        # Process pairs in parallel with progress bar
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_cosine_for_pair)(pair_data) 
            for pair_data in tqdm.tqdm(pairs, desc="Cosine Similarity")
        )
        
        print(f"Completed cosine similarity calculation in {time.time() - start_time:.2f} seconds")
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        return True
    
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return False

def _calculate_edit_for_pair(pair_data):
    """Calculate Edit Distance similarity for a single pair."""
    file_contents, file1, file2, i, j, max_chars = pair_data
    
    if i == j:
        similarity = 1.0
    else:
        # Limit text length to improve performance
        text1 = file_contents[file1][:max_chars]
        text2 = file_contents[file2][:max_chars]
        similarity = textdistance.levenshtein.normalized_similarity(text1, text2)
    
    return {'file1': file1, 'file2': file2, 'similarity': similarity}

def calculate_edit_similarity(file_contents, output_csv, n_jobs, max_chars=500, sample_ratio=1.0):
    """Calculate Edit Distance Similarity with parallelization and sampling."""
    file_names = list(file_contents.keys())
    n_files = len(file_names)
    print(f"Processing {n_files} files for Edit Distance similarity using {n_jobs} CPU cores...")
    print(f"Limiting to first {max_chars} characters per file for edit distance calculation")
    
    # Generate all possible pairs
    all_pairs = []
    for i, file1 in enumerate(file_names):
        for j in range(i, len(file_names)):
            file2 = file_names[j]
            all_pairs.append((i, j, file1, file2))
    
    # Sample pairs if requested
    total_pairs = len(all_pairs)
    sample_size = max(int(total_pairs * sample_ratio), 1)  # At least 1 pair
    
    # Always include all self-comparisons (i==j) in the sample
    self_pairs = [(i, i, file_names[i], file_names[i]) for i in range(n_files)]
    
    if sample_ratio < 1.0 and sample_size < total_pairs:
        # Set random seed for reproducibility
        random.seed(42)
        
        # Get non-self pairs for sampling
        non_self_pairs = [p for p in all_pairs if p[0] != p[1]]
        
        # Calculate how many non-self pairs to sample
        non_self_sample_size = min(sample_size - len(self_pairs), len(non_self_pairs))
        
        # Sample non-self pairs
        sampled_non_self = random.sample(non_self_pairs, non_self_sample_size)
        
        # Combine with self pairs
        sampled_pairs = self_pairs + sampled_non_self
        
        print(f"Sampling {len(sampled_pairs)} document pairs ({sample_ratio:.1%} of {total_pairs})")
        print(f"  - {len(self_pairs)} self comparisons (always included)")
        print(f"  - {len(sampled_non_self)} sampled non-self comparisons")
    else:
        sampled_pairs = all_pairs
        print(f"Calculating all {total_pairs} document pairs")
    
    # Convert to format needed for calculation
    pairs = [(file_contents, file1, file2, i, j, max_chars) for i, j, file1, file2 in sampled_pairs]
    
    start_time = time.time()
    
    # Process in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_edit_for_pair)(pair_data) 
        for pair_data in tqdm.tqdm(pairs, desc="Edit Distance")
    )
    
    print(f"Completed edit distance calculation in {time.time() - start_time:.2f} seconds")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return True

def calculate_all_similarities(input_dir, results_dir, n_jobs=None, max_edit_chars=500, 
                              skip_existing=True, max_files=0, sample_ratio=1.0, 
                              skip_edit=False):
    """Calculate all similarity metrics with optimizations.
    
    Args:
        input_dir: Directory containing text files
        results_dir: Directory to store results
        n_jobs: Number of CPU cores to use (None = auto)
        max_edit_chars: Maximum characters to consider for edit distance
        skip_existing: Skip calculations if output files already exist
        max_files: Maximum number of files to process (0 = all files)
        sample_ratio: Ratio of document pairs to sample (0.0-1.0)
        skip_edit: Skip edit distance calculation entirely
    """
    # Determine number of CPU cores to use
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Create descriptive file suffixes
    suffix = ""
    if sample_ratio < 1.0:
        suffix += f"_sample{int(sample_ratio*100)}pct"
    if max_files > 0:
        suffix += f"_max{max_files}files"
    
    # Output file paths
    jaccard_csv = os.path.join(results_dir, f"jaccard_similarity{suffix}.csv")
    cosine_csv = os.path.join(results_dir, f"cosine_similarity{suffix}.csv")
    edit_csv = os.path.join(results_dir, f"edit_similarity{suffix}.csv")
    
    # Load files with potential max_files limit
    file_contents = load_files(input_dir, max_files)
    
    print(f"\nPerforming similarity calculations with:")
    print(f"- {len(file_contents)} files")
    print(f"- {n_jobs} CPU cores")
    print(f"- {sample_ratio:.1%} sampling ratio for document pairs")
    
    # Jaccard similarity
    if not os.path.exists(jaccard_csv) or not skip_existing:
        print("\nComputing Jaccard similarity...")
        calculate_jaccard_similarity(file_contents, jaccard_csv, n_jobs, sample_ratio)
    else:
        print(f"\nSkipping Jaccard similarity (file exists: {jaccard_csv})")
    
    # Cosine similarity
    if not os.path.exists(cosine_csv) or not skip_existing:
        print("\nComputing Cosine similarity...")
        calculate_cosine_similarity(file_contents, cosine_csv, n_jobs, sample_ratio)
    else:
        print(f"\nSkipping Cosine similarity (file exists: {cosine_csv})")
    
    # Edit distance similarity
    if skip_edit:
        print("\nSkipping Edit Distance similarity (disabled by user)")
        # Create empty file if it doesn't exist
        if not os.path.exists(edit_csv):
            pd.DataFrame(columns=['file1', 'file2', 'similarity']).to_csv(edit_csv, index=False)
    elif not os.path.exists(edit_csv) or not skip_existing:
        print("\nComputing Edit Distance similarity...")
        calculate_edit_similarity(file_contents, edit_csv, n_jobs, max_chars=max_edit_chars, sample_ratio=sample_ratio)
    else:
        print(f"\nSkipping Edit Distance similarity (file exists: {edit_csv})")
    
    return jaccard_csv, cosine_csv, edit_csv

# For testing the module directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate document similarities with optimizations.')
    parser.add_argument('--input-dir', default="haunted_places_files", help='Directory with text files')
    parser.add_argument('--results-dir', default="similarity_results", help='Directory for results')
    parser.add_argument('--jobs', type=int, default=0, help='Number of CPU cores (0 = auto)')
    parser.add_argument('--max-files', type=int, default=0, help='Max number of files to process (0 = all)')
    parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio for document pairs (0.0-1.0)')
    parser.add_argument('--edit-chars', type=int, default=500, help='Max chars for edit distance')
    parser.add_argument('--skip-edit', action='store_true', help='Skip edit distance calculation')
    parser.add_argument('--force', action='store_true', help='Force recalculation of existing results')
    
    args = parser.parse_args()
    
    # Make sure results directory exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Set number of CPU cores to use
    n_jobs = args.jobs if args.jobs > 0 else None
    
    # Run similarity calculations
    calculate_all_similarities(
        args.input_dir, 
        args.results_dir, 
        n_jobs=n_jobs,
        max_edit_chars=args.edit_chars,
        skip_existing=not args.force,
        max_files=args.max_files,
        sample_ratio=args.sample,
        skip_edit=args.skip_edit
    )