import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import os

def load_similarity_data():
    """Load the similarity CSV files"""
    base_dir = "similarity_results"
    jaccard_df = pd.read_csv(os.path.join(base_dir, "jaccard_similarity.csv"))
    cosine_df = pd.read_csv(os.path.join(base_dir, "cosine_similarity.csv"))
    edit_df = pd.read_csv(os.path.join(base_dir, "edit_similarity.csv"))
    
    return jaccard_df, cosine_df, edit_df

def create_similarity_matrix(df):
    """Convert the file1, file2, similarity format to a matrix"""
    # Get unique filenames
    files = sorted(list(set(df['file1'].unique()) | set(df['file2'].unique())))
    n_files = len(files)
    
    # Create a mapping from filename to index
    file_to_idx = {file: idx for idx, file in enumerate(files)}
    
    # Initialize similarity matrix
    sim_matrix = np.zeros((n_files, n_files))
    
    # Fill the similarity matrix
    for _, row in df.iterrows():
        i = file_to_idx[row['file1']]
        j = file_to_idx[row['file2']]
        sim_matrix[i, j] = row['similarity']
        sim_matrix[j, i] = row['similarity']  # Make it symmetric
    
    return sim_matrix, files

def convert_to_distance(similarity_matrix):
    """Convert similarity matrix to distance matrix"""
    # For similarity values between 0-1, distance = 1 - similarity
    return 1 - similarity_matrix

def perform_clustering(distance_matrix, n_clusters=5):
    """Perform hierarchical clustering on the distance matrix"""
    try:
        # Try with newer scikit-learn API
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', 
            linkage='average'
        )
    except TypeError:
        # Fall back to older scikit-learn API
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',  # older versions use 'metric' instead of 'affinity'
            linkage='average'
        )
    
    clusters = clustering.fit_predict(distance_matrix)
    return clusters

def visualize_clusters(distance_matrix, clusters, files, title):
    """Visualize clusters using MDS for dimensionality reduction"""
    # Use MDS to project the distance matrix into 2D space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)
    
    # Create a dataframe for plotting
    cluster_df = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'cluster': clusters,
        'filename': [os.path.basename(f) for f in files]
    })
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=cluster_df, x='x', y='y', hue='cluster', palette='viridis', s=100)
    
    # Add labels to points
    for i, row in cluster_df.iterrows():
        plt.text(row['x']+0.02, row['y']+0.02, row['filename'].replace('.txt', ''), 
                 fontsize=8, alpha=0.7)
    
    plt.title(f'Document Clusters using {title}')
    plt.tight_layout()
    plt.savefig(f"cluster_visualization_{title.replace(' ', '_')}.png")
    plt.close()
    
    return cluster_df

def compare_clusters(jaccard_clusters, cosine_clusters, edit_clusters, files):
    """Compare cluster assignments between different similarity measures"""
    comparison_df = pd.DataFrame({
        'filename': [os.path.basename(f) for f in files],
        'jaccard_cluster': jaccard_clusters,
        'cosine_cluster': cosine_clusters,
        'edit_cluster': edit_clusters
    })
    
    # Calculate agreement percentages
    total = len(comparison_df)
    jaccard_cosine_agree = sum(comparison_df['jaccard_cluster'] == comparison_df['cosine_cluster']) / total
    jaccard_edit_agree = sum(comparison_df['jaccard_cluster'] == comparison_df['edit_cluster']) / total
    cosine_edit_agree = sum(comparison_df['cosine_cluster'] == comparison_df['edit_cluster']) / total
    
    print(f"Agreement between Jaccard and Cosine clusters: {jaccard_cosine_agree:.2%}")
    print(f"Agreement between Jaccard and Edit Distance clusters: {jaccard_edit_agree:.2%}")
    print(f"Agreement between Cosine and Edit Distance clusters: {cosine_edit_agree:.2%}")
    
    # Visualize cluster comparison with heatmap
    plt.figure(figsize=(12, 8))
    cluster_counts = pd.crosstab(
        [comparison_df['jaccard_cluster'], comparison_df['cosine_cluster']], 
        comparison_df['edit_cluster']
    )
    sns.heatmap(cluster_counts, annot=True, cmap='Blues', fmt='d')
    plt.title('Comparison of Cluster Assignments Across Methods')
    plt.xlabel('Edit Distance Clusters')
    plt.ylabel('Jaccard (outer) / Cosine (inner) Clusters')
    plt.tight_layout()
    plt.savefig("cluster_comparison.png")
    plt.close()
    
    return comparison_df

def analyze_differences(jaccard_matrix, cosine_matrix, edit_matrix, files, n_examples=5):
    """Analyze differences in how each similarity measure scores document pairs"""
    pairs = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file1 = os.path.basename(files[i])
            file2 = os.path.basename(files[j])
            pairs.append({
                'file1': file1,
                'file2': file2,
                'jaccard': jaccard_matrix[i, j],
                'cosine': cosine_matrix[i, j],
                'edit': edit_matrix[i, j],
                'j_minus_c': jaccard_matrix[i, j] - cosine_matrix[i, j],
                'j_minus_e': jaccard_matrix[i, j] - edit_matrix[i, j],
                'c_minus_e': cosine_matrix[i, j] - edit_matrix[i, j]
            })
    
    diff_df = pd.DataFrame(pairs)
    
    # Find examples with the largest differences
    print("\nLargest differences between similarity measures:")
    
    print("\nJaccard vs. Cosine (most extreme examples):")
    for _, row in diff_df.nlargest(n_examples, 'j_minus_c').iterrows():
        print(f"{row['file1']} and {row['file2']}: Jaccard={row['jaccard']:.3f}, Cosine={row['cosine']:.3f}, Diff={row['j_minus_c']:.3f}")
    
    print("\nJaccard vs. Edit Distance (most extreme examples):")
    for _, row in diff_df.nlargest(n_examples, 'j_minus_e').iterrows():
        print(f"{row['file1']} and {row['file2']}: Jaccard={row['jaccard']:.3f}, Edit={row['edit']:.3f}, Diff={row['j_minus_e']:.3f}")
    
    print("\nCosine vs. Edit Distance (most extreme examples):")
    for _, row in diff_df.nlargest(n_examples, 'c_minus_e').iterrows():
        print(f"{row['file1']} and {row['file2']}: Cosine={row['cosine']:.3f}, Edit={row['edit']:.3f}, Diff={row['c_minus_e']:.3f}")
    
    # Create correlation matrix
    corr_matrix = np.corrcoef([diff_df['jaccard'], diff_df['cosine'], diff_df['edit']])
    print("\nCorrelation between similarity measures:")
    print(f"Jaccard-Cosine: {corr_matrix[0,1]:.3f}")
    print(f"Jaccard-Edit: {corr_matrix[0,2]:.3f}")
    print(f"Cosine-Edit: {corr_matrix[1,2]:.3f}")
    
    return diff_df

def main():
    print("Loading similarity data...")
    jaccard_df, cosine_df, edit_df = load_similarity_data()
    
    print("Processing Jaccard similarity...")
    jaccard_matrix, files = create_similarity_matrix(jaccard_df)
    jaccard_dist = convert_to_distance(jaccard_matrix)
    
    print("Processing Cosine similarity...")
    cosine_matrix, _ = create_similarity_matrix(cosine_df)
    cosine_dist = convert_to_distance(cosine_matrix)
    
    print("Processing Edit Distance similarity...")
    edit_matrix, _ = create_similarity_matrix(edit_df)
    edit_dist = convert_to_distance(edit_matrix)
    
    # Determine optimal number of clusters
    n_clusters = 5  # You can use silhouette scores to find optimal number
    
    print(f"Performing clustering with {n_clusters} clusters...")
    jaccard_clusters = perform_clustering(jaccard_dist, n_clusters)
    cosine_clusters = perform_clustering(cosine_dist, n_clusters)
    edit_clusters = perform_clustering(edit_dist, n_clusters)
    
    print("Visualizing clusters...")
    jaccard_viz = visualize_clusters(jaccard_dist, jaccard_clusters, files, "Jaccard Similarity")
    cosine_viz = visualize_clusters(cosine_dist, cosine_clusters, files, "Cosine Similarity")
    edit_viz = visualize_clusters(edit_dist, edit_clusters, files, "Edit Distance Similarity")
    
    print("Comparing clusters...")
    comparison = compare_clusters(jaccard_clusters, cosine_clusters, edit_clusters, files)
    
    print("Analyzing differences...")
    diff_analysis = analyze_differences(jaccard_matrix, cosine_matrix, edit_matrix, files)
    
    print("\nAnalysis summary:")
    print("""
    Key differences between similarity measures:
    
    1. Jaccard similarity focuses on shared tokens (words) and their presence/absence
       without considering word frequency or position
    
    2. Cosine similarity uses word frequency information (TF-IDF), making it sensitive
       to important terms and less affected by common words
    
    3. Edit distance similarity considers the sequence and position of text,
       making it useful for detecting small edits and structural similarities
    
    These differences explain why documents might be clustered differently by each method:
    - Text with similar vocabulary but different structures cluster together in Jaccard
    - Documents about similar topics cluster together in Cosine 
    - Documents with similar structures cluster together in Edit Distance
    """)

if __name__ == "__main__":
    main()