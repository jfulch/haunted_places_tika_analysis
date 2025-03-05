import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def create_heatmap(df, title, output_path):
    """Create a heatmap visualization of similarity data."""
    # Create a pivot table for the heatmap
    pivot_df = df.pivot(index='file1', columns='file2', values='similarity')
    
    # Create a mask for the upper triangle (to avoid redundancy)
    mask = np.triu(np.ones_like(pivot_df, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(pivot_df, mask=mask, cmap=cmap, vmin=0, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title(f"{title} Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def create_mds_plot(df, title, output_path):
    """Create MDS plot to visualize document similarity in 2D space."""
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
    
    # Convert to distance matrix
    distance_matrix = 1 - sim_matrix
    
    # Use MDS to project the distance matrix into 2D space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    plt.scatter(positions[:, 0], positions[:, 1], s=100)
    
    # Add labels to the points
    for i, file in enumerate(files):
        short_name = os.path.basename(file)[:15]  # truncate long filenames
        plt.annotate(short_name, (positions[i, 0], positions[i, 1]), 
                     fontsize=8, alpha=0.7)
    
    plt.title(f"{title} - Document Similarity in 2D Space")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def create_histogram(df, title, output_path):
    """Create histogram of similarity scores."""
    plt.figure(figsize=(10, 6))
    
    # Filter out self-comparisons
    non_self_df = df[df['file1'] != df['file2']]
    
    # Create histogram
    sns.histplot(data=non_self_df, x='similarity', bins=20, kde=True)
    
    plt.title(f"Distribution of {title} Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_markdown_report(jaccard_csv, cosine_csv, edit_csv):
    """Generate a comprehensive markdown report of similarity analysis."""
    print("Generating markdown report...")
    
    # Load the similarity data
    jaccard_df = pd.read_csv(jaccard_csv)
    cosine_df = pd.read_csv(cosine_csv)
    edit_df = pd.read_csv(edit_csv)
    
    # Create output directory for visualizations
    results_dir = os.path.dirname(jaccard_csv)
    viz_dir = os.path.join(results_dir, "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Heatmaps
    jaccard_heatmap = create_heatmap(jaccard_df, "Jaccard Similarity", os.path.join(viz_dir, "jaccard_heatmap.png"))
    cosine_heatmap = create_heatmap(cosine_df, "Cosine Similarity", os.path.join(viz_dir, "cosine_heatmap.png"))
    edit_heatmap = create_heatmap(edit_df, "Edit Distance Similarity", os.path.join(viz_dir, "edit_heatmap.png"))
    
    # MDS plots
    jaccard_mds = create_mds_plot(jaccard_df, "Jaccard Similarity", os.path.join(viz_dir, "jaccard_mds.png"))
    cosine_mds = create_mds_plot(cosine_df, "Cosine Similarity", os.path.join(viz_dir, "cosine_mds.png"))
    edit_mds = create_mds_plot(edit_df, "Edit Distance Similarity", os.path.join(viz_dir, "edit_mds.png"))
    
    # Histograms
    jaccard_hist = create_histogram(jaccard_df, "Jaccard", os.path.join(viz_dir, "jaccard_histogram.png"))
    cosine_hist = create_histogram(cosine_df, "Cosine", os.path.join(viz_dir, "cosine_histogram.png"))
    edit_hist = create_histogram(edit_df, "Edit Distance", os.path.join(viz_dir, "edit_histogram.png"))
    
    # Prepare markdown content
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Document Similarity Analysis Report
Generated on: {now}

## Overview
This report compares three different similarity metrics for document comparison:
- **Jaccard Similarity**: Measures the overlap between word sets
- **Cosine Similarity**: Measures the angular similarity between TF-IDF vector representations
- **Edit Distance Similarity**: Measures the character-level editing operations required to transform one text into another

## Data Files Used
- Input files: [{os.path.abspath(os.path.dirname(results_dir))}/haunted_places_files](file://{os.path.abspath(os.path.dirname(results_dir))}/haunted_places_files)
- Similarity results:
  - [Jaccard Similarity CSV]({os.path.relpath(jaccard_csv, results_dir)})
  - [Cosine Similarity CSV]({os.path.relpath(cosine_csv, results_dir)})
  - [Edit Distance Similarity CSV]({os.path.relpath(edit_csv, results_dir)})

## Similarity Metrics Summary

### Jaccard Similarity
- Based on token overlap (word sets)
- Range: 0 (no overlap) to 1 (identical sets)
- Ignores word frequency and order
- Best for: Comparing document vocabulary regardless of structure

### Cosine Similarity 
- Based on TF-IDF vector representation
- Range: 0 (completely different) to 1 (identical direction)
- Considers term frequency and importance
- Best for: Topic-based similarity and content comparison

### Edit Distance Similarity
- Based on character-level Levenshtein distance
- Range: 0 (completely different) to 1 (identical texts)
- Sensitive to spelling, structure, and word order
- Best for: Detecting small edits and structural similarities

## Similarity Distributions

### Jaccard Similarity Distribution
![Jaccard Similarity Histogram]({os.path.relpath(jaccard_hist, results_dir)})

### Cosine Similarity Distribution
![Cosine Similarity Histogram]({os.path.relpath(cosine_hist, results_dir)})

### Edit Distance Similarity Distribution
![Edit Distance Similarity Histogram]({os.path.relpath(edit_hist, results_dir)})

## Document Similarity Maps

These visualizations show documents positioned in 2D space based on their similarity. 
Documents that are similar to each other appear closer together.

### Jaccard Similarity Map
![Jaccard MDS Plot]({os.path.relpath(jaccard_mds, results_dir)})

### Cosine Similarity Map
![Cosine MDS Plot]({os.path.relpath(cosine_mds, results_dir)})

### Edit Distance Similarity Map
![Edit Distance MDS Plot]({os.path.relpath(edit_mds, results_dir)})

## Similarity Heatmaps

These heatmaps visualize the similarity between each pair of documents. 
Darker colors indicate higher similarity.

### Jaccard Similarity Heatmap
![Jaccard Heatmap]({os.path.relpath(jaccard_heatmap, results_dir)})

### Cosine Similarity Heatmap
![Cosine Heatmap]({os.path.relpath(cosine_heatmap, results_dir)})

### Edit Distance Similarity Heatmap
![Edit Heatmap]({os.path.relpath(edit_heatmap, results_dir)})

## Analysis Results

"""

    # Add similarity statistics
    for metric_name, df in [("Jaccard", jaccard_df), ("Cosine", cosine_df), ("Edit Distance", edit_df)]:
        # Calculate statistics excluding self-comparisons (where similarity = 1.0 because file1 = file2)
        non_self_df = df[df['file1'] != df['file2']]
        
        if len(non_self_df) > 0:
            avg_sim = non_self_df['similarity'].mean()
            med_sim = non_self_df['similarity'].median()
            min_sim = non_self_df['similarity'].min()
            max_sim = non_self_df['similarity'].max()
            
            markdown += f"""### {metric_name} Similarity Statistics
- **Average similarity**: {avg_sim:.4f}
- **Median similarity**: {med_sim:.4f}
- **Minimum similarity**: {min_sim:.4f}
- **Maximum similarity**: {max_sim:.4f}

"""
    
    # Find interesting document pairs (most similar and most different)
    markdown += "## Notable Document Comparisons\n\n"
    
    for metric_name, df in [("Jaccard", jaccard_df), ("Cosine", cosine_df), ("Edit Distance", edit_df)]:
        non_self_df = df[df['file1'] != df['file2']]
        
        if len(non_self_df) > 0:
            # Most similar pair
            most_similar = non_self_df.nlargest(1, 'similarity').iloc[0]
            most_similar_val = most_similar['similarity']
            file1 = most_similar['file1']
            file2 = most_similar['file2']
            
            # Least similar pair
            least_similar = non_self_df.nsmallest(1, 'similarity').iloc[0]
            least_similar_val = least_similar['similarity']
            file3 = least_similar['file1']
            file4 = least_similar['file2']
            
            file1_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", file1)
            file2_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", file2)
            file3_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", file3)
            file4_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", file4)
            
            markdown += f"""### {metric_name} Similarity
- **Most similar pair**: [`{file1}`](file://{os.path.abspath(file1_path)}) and [`{file2}`](file://{os.path.abspath(file2_path)}) (similarity: {most_similar_val:.4f})
- **Least similar pair**: [`{file3}`](file://{os.path.abspath(file3_path)}) and [`{file4}`](file://{os.path.abspath(file4_path)}) (similarity: {least_similar_val:.4f})

"""
    
    # Create correlation heatmap
    print("Generating correlation heatmap...")
    # Merge dataframes to compare metrics
    merged_df = pd.merge(
        pd.merge(
            jaccard_df, 
            cosine_df, 
            on=['file1', 'file2'], 
            suffixes=('_jaccard', '_cosine')
        ),
        edit_df,
        on=['file1', 'file2']
    ).rename(columns={'similarity': 'similarity_edit'})
    
    # Calculate correlations between metrics
    corr = merged_df[['similarity_jaccard', 'similarity_cosine', 'similarity_edit']].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
    plt.title('Correlation Between Similarity Metrics')
    correlation_path = os.path.join(viz_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(correlation_path)
    plt.close()
    
    markdown += "## Cross-Metric Comparison\n\n"
    markdown += f"![Correlation Between Metrics]({os.path.relpath(correlation_path, results_dir)})\n\n"
    
    markdown += f"""### Correlation Between Metrics
|               | Jaccard | Cosine | Edit Distance |
|---------------|---------|--------|--------------|
| Jaccard       | 1.000   | {corr.loc['similarity_jaccard', 'similarity_cosine']:.4f} | {corr.loc['similarity_jaccard', 'similarity_edit']:.4f} |
| Cosine        | {corr.loc['similarity_cosine', 'similarity_jaccard']:.4f} | 1.000   | {corr.loc['similarity_cosine', 'similarity_edit']:.4f} |
| Edit Distance | {corr.loc['similarity_edit', 'similarity_jaccard']:.4f} | {corr.loc['similarity_edit', 'similarity_cosine']:.4f} | 1.000   |

"""
    
    # Find document pairs with biggest differences between metrics
    merged_df['jaccard_cosine_diff'] = abs(merged_df['similarity_jaccard'] - merged_df['similarity_cosine'])
    merged_df['jaccard_edit_diff'] = abs(merged_df['similarity_jaccard'] - merged_df['similarity_edit'])
    merged_df['cosine_edit_diff'] = abs(merged_df['similarity_cosine'] - merged_df['similarity_edit'])
    
    # Get top 3 discrepancies
    top_jc_diff = merged_df.nlargest(3, 'jaccard_cosine_diff')
    top_je_diff = merged_df.nlargest(3, 'jaccard_edit_diff')
    top_ce_diff = merged_df.nlargest(3, 'cosine_edit_diff')
    
    markdown += "### Largest Discrepancies Between Metrics\n\n"
    
    markdown += "#### Jaccard vs. Cosine\n"
    for _, row in top_jc_diff.iterrows():
        file1_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", row['file1'])
        file2_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", row['file2'])
        markdown += f"- [`{row['file1']}`](file://{os.path.abspath(file1_path)}) and [`{row['file2']}`](file://{os.path.abspath(file2_path)}): Jaccard={row['similarity_jaccard']:.4f}, Cosine={row['similarity_cosine']:.4f}, Difference={row['jaccard_cosine_diff']:.4f}\n"
    
    markdown += "\n#### Jaccard vs. Edit Distance\n"
    for _, row in top_je_diff.iterrows():
        file1_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", row['file1'])
        file2_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", row['file2'])
        markdown += f"- [`{row['file1']}`](file://{os.path.abspath(file1_path)}) and [`{row['file2']}`](file://{os.path.abspath(file2_path)}): Jaccard={row['similarity_jaccard']:.4f}, Edit={row['similarity_edit']:.4f}, Difference={row['jaccard_edit_diff']:.4f}\n"
    
    markdown += "\n#### Cosine vs. Edit Distance\n"
    for _, row in top_ce_diff.iterrows():
        file1_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", row['file1'])
        file2_path = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "haunted_places_files", row['file2'])
        markdown += f"- [`{row['file1']}`](file://{os.path.abspath(file1_path)}) and [`{row['file2']}`](file://{os.path.abspath(file2_path)}): Cosine={row['similarity_cosine']:.4f}, Edit={row['similarity_edit']:.4f}, Difference={row['cosine_edit_diff']:.4f}\n"
    
    markdown += """
## Conclusions

### Key Observations
1. **Jaccard Similarity** focuses on shared vocabulary without considering word frequency or position.
   - Documents with similar word sets but different contexts may appear similar.

2. **Cosine Similarity** considers term frequencies and importance.
   - Better at identifying topical similarity regardless of text length.
   - Less influenced by common words that appear in many documents.

3. **Edit Distance Similarity** is sensitive to the sequence and position of text.
   - Identifies documents with similar structure and phrasing.
   - Can detect small edits and variations in text.

### Which Metric to Choose?
- **For topic similarity**: Cosine similarity is generally the best choice.
- **For structural similarity**: Edit distance is more appropriate.
- **For vocabulary overlap**: Jaccard similarity works well.

The choice of similarity metric should depend on the specific requirements of your analysis.

## Additional Resources
- [Understanding Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)
- [Understanding Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Understanding Edit Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Multidimensional Scaling Explained](https://en.wikipedia.org/wiki/Multidimensional_scaling)
"""
    
    # Save markdown to file
    markdown_path = os.path.join(results_dir, "similarity_analysis_report.md")
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"Markdown report saved to {markdown_path}")
    return markdown_path