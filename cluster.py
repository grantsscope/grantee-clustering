import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the embeddings
embeddings = np.load('project_embeddings.npy')

# Reduce dimensionality with t-SNE
tsne_embeddings = TSNE().fit_transform(embeddings)

# Clustering with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=1)  # Adjust parameters as needed
cluster_labels = clusterer.fit_predict(tsne_embeddings)
cluster_probabilities = clusterer.probabilities_

# Load the original projects data
original_data_file = 'GG18 - Climate - Approved.csv'  # Replace with your file path
original_df = pd.read_csv(original_data_file)

# Add cluster labels and probabilities to the original data
original_df['Cluster'] = cluster_labels
original_df['Cluster_Probability'] = cluster_probabilities
original_df['TSNE_X'] = tsne_embeddings[:, 0]
original_df['TSNE_Y'] = tsne_embeddings[:, 1]

# Visualization
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=cluster_labels, cmap='Spectral', s=50)
for i, point in original_df.iterrows():
    plt.annotate(point['Title'], (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), fontsize=12, ha='right', va='bottom')
plt.title('Project Clusters with t-SNE and HDBSCAN')
plt.colorbar()
plt.show()


# Save the data with cluster labels, probabilities, and t-SNE coordinates
original_df.to_csv('projects_with_clusters_and_tsne.csv', index=False)
