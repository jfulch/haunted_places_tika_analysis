# Document Similarity Analysis Report
Generated on: 2025-03-06 21:15:16

## Overview
This report compares three different similarity metrics for document comparison:
- **Jaccard Similarity**: Measures the overlap between word sets
- **Cosine Similarity**: Measures the angular similarity between TF-IDF vector representations
- **Edit Distance Similarity**: Measures the character-level editing operations required to transform one text into another

## Data Files Used
- Input files: [/Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files)
- Similarity results:
  - [Jaccard Similarity CSV](jaccard_similarity.csv)
  - [Cosine Similarity CSV](cosine_similarity.csv)
  - [Edit Distance Similarity CSV](edit_similarity.csv)

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
![Jaccard Similarity Histogram](visualizations/jaccard_histogram.png)

### Cosine Similarity Distribution
![Cosine Similarity Histogram](visualizations/cosine_histogram.png)

### Edit Distance Similarity Distribution
![Edit Distance Similarity Histogram](visualizations/edit_histogram.png)

## Document Similarity Maps

These visualizations show documents positioned in 2D space based on their similarity. 
Documents that are similar to each other appear closer together.

### Jaccard Similarity Map
![Jaccard MDS Plot](visualizations/jaccard_mds.png)

### Cosine Similarity Map
![Cosine MDS Plot](visualizations/cosine_mds.png)

### Edit Distance Similarity Map
![Edit Distance MDS Plot](visualizations/edit_mds.png)

## Similarity Heatmaps

These heatmaps visualize the similarity between each pair of documents. 
Darker colors indicate higher similarity.

### Jaccard Similarity Heatmap
![Jaccard Heatmap](visualizations/jaccard_heatmap.png)

### Cosine Similarity Heatmap
![Cosine Heatmap](visualizations/cosine_heatmap.png)

### Edit Distance Similarity Heatmap
![Edit Heatmap](visualizations/edit_heatmap.png)

## Analysis Results

### Jaccard Similarity Statistics
- **Average similarity**: 0.8415
- **Median similarity**: 0.8868
- **Minimum similarity**: 0.5753
- **Maximum similarity**: 0.9800

### Cosine Similarity Statistics
- **Average similarity**: 0.7062
- **Median similarity**: 0.7781
- **Minimum similarity**: 0.1284
- **Maximum similarity**: 0.9911

### Edit Distance Similarity Statistics
- **Average similarity**: 0.9215
- **Median similarity**: 0.9476
- **Minimum similarity**: 0.7488
- **Maximum similarity**: 0.9994

## Notable Document Comparisons

### Jaccard Similarity
- **Most similar pair**: [`place_95.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_95.txt) and [`place_100.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_100.txt) (similarity: 0.9800)
- **Least similar pair**: [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt) and [`place_101.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_101.txt) (similarity: 0.5753)

### Cosine Similarity
- **Most similar pair**: [`place_64.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_64.txt) and [`place_63.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_63.txt) (similarity: 0.9911)
- **Least similar pair**: [`place_98.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_98.txt) and [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt) (similarity: 0.1284)

### Edit Distance Similarity
- **Most similar pair**: [`place_64.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_64.txt) and [`place_65.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_65.txt) (similarity: 0.9994)
- **Least similar pair**: [`place_41.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_41.txt) and [`place_61.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_61.txt) (similarity: 0.7488)

## Cross-Metric Comparison

![Correlation Between Metrics](visualizations/correlation_heatmap.png)

### Correlation Between Metrics
|               | Jaccard | Cosine | Edit Distance |
|---------------|---------|--------|--------------|
| Jaccard       | 1.000   | 0.9533 | 0.9693 |
| Cosine        | 0.9533 | 1.000   | 0.9541 |
| Edit Distance | 0.9693 | 0.9541 | 1.000   |

### Largest Discrepancies Between Metrics

#### Jaccard vs. Cosine
- [`place_22.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_22.txt) and [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt): Jaccard=0.8036, Cosine=0.3179, Difference=0.4857
- [`place_41.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_41.txt) and [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt): Jaccard=0.8182, Cosine=0.3494, Difference=0.4688
- [`place_53.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_53.txt) and [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt): Jaccard=0.7857, Cosine=0.3289, Difference=0.4568

#### Jaccard vs. Edit Distance
- [`place_28.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_28.txt) and [`place_60.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_60.txt): Jaccard=0.6087, Edit=0.8219, Difference=0.2132
- [`place_86.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_86.txt) and [`place_60.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_60.txt): Jaccard=0.6087, Edit=0.8193, Difference=0.2106
- [`place_52.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_52.txt) and [`place_60.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_60.txt): Jaccard=0.6176, Edit=0.8283, Difference=0.2106

#### Cosine vs. Edit Distance
- [`place_98.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_98.txt) and [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt): Cosine=0.1284, Edit=0.7743, Difference=0.6459
- [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt) and [`place_101.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_101.txt): Cosine=0.1391, Edit=0.7706, Difference=0.6315
- [`place_42.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_42.txt) and [`place_7.txt`](file:///Users/jfulch/git/school/haunted_places_tika_analysis/haunted_places_files/place_7.txt): Cosine=0.1520, Edit=0.7660, Difference=0.6140

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
