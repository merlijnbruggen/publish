from sklearn.manifold import MDS
import numpy as np
from sklearn.metrics import pairwise_distances
import the_worm
import matplotlib.pyplot as plt
import os
import scipy.linalg
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


# Path to the directory containing your files
L1 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L1'
L1_Late = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_LateL1'
L2 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L2'
L3 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L3'
L4 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L4'
Adult = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_Adult'

directory_list = [L1, L1_Late, L2, L3, L4]

'''
# List to store distance matrices
all_distance_matrices = []

for directory in directory_list:
    distance_matrix, _ = the_worm.compute_within_cluster_distances(directory, the_worm.calculate_covariance_matrix)
    all_distance_matrices.append(distance_matrix)

# Step 1: Compute representative points for each cluster (centroid)
cluster_centroids = []
for distance_matrix in all_distance_matrices:
    centroid = np.mean(distance_matrix, axis=0)  # Assuming rows represent data points
    cluster_centroids.append(centroid)

# Step 2: Ensure all centroids have the same length
max_length = max(len(centroid) for centroid in cluster_centroids)
for i, centroid in enumerate(cluster_centroids):
    if len(centroid) < max_length:
        cluster_centroids[i] = np.pad(centroid, (0, max_length - len(centroid)), mode='constant')

# Step 3: Compute distance matrix between cluster centroids
centroid_distances = pairwise_distances(cluster_centroids, metric='euclidean')

# Visualize within-cluster distance matrices
plt.figure(figsize=(12, 4))
for i, distance_matrix in enumerate(all_distance_matrices):
    plt.subplot(1, len(all_distance_matrices), i+1)
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f'Cluster {i+1}')
    plt.xlabel('Worm #')
    plt.ylabel('Worm #')
plt.suptitle('Within-Cluster Distance Matrices')
plt.tight_layout()
plt.show()

# Visualize between-cluster distance matrix
plt.figure()
plt.imshow(centroid_distances, cmap='viridis')
plt.colorbar()
plt.title('Between-Cluster Distance Matrix (uses mean)')
plt.xlabel('Developmental stage ')
plt.ylabel('Developmental stage')
plt.show()
'''
      

# IMPORTANT to note is that Kmeans uses the euclidean distance, and the cov matrices don't live in euclidean space

def apply_kmeans(covariance_matrices, num_clusters):
    """
    Apply k-means clustering algorithm to a dataset of covariance matrices.

    Parameters:
    - covariance_matrices (list of 2D arrays): Array of covariance matrices.
    - num_clusters (int): Number of clusters to create.

    Returns:
    - labels (array): Array of cluster labels assigned to each covariance matrix.
    """

    # Flatten each covariance matrix into a 1D array
    flattened_matrices = [cov.flatten() for cov in covariance_matrices]

    # Initialize and fit k-means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(flattened_matrices)

    # Get cluster labels
    labels = kmeans.labels_
    print(labels)
    return labels

def calculate_wcss(data, max_clusters):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Load all covariance matrices
all_covariance_matrices = []
for directory in directory_list:
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_path = os.path.join(directory, filename)
            covariance_matrix = the_worm.calculate_covariance_matrix(file_path)
            all_covariance_matrices.append(covariance_matrix)
        if filename == '102122_N2_adult_crawl_0001.txt':
            print("Error: reached file I couldn't read")
            break

def compute_wcss_plot(data, max_clusters=10):
    """
    Compute the Within-Cluster Sum of Squares (WCSS) plot for a dataset.

    Parameters:
    - data: Numpy array containing the dataset. Each element should be a 2D array.
    - max_clusters: Maximum number of clusters to consider.

    Returns:
    - wcss_values: List containing the WCSS values for each number of clusters.
    """

    wcss_values = []

    # Flatten each 2D array into a 1D array
    flattened_data = [matrix.flatten() for matrix in data]

    # Convert the list of flattened matrices into a 2D array
    flattened_data = np.array(flattened_data)

    # Compute WCSS for different number of clusters
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(flattened_data)
        wcss_values.append(kmeans.inertia_)

    # Plot the WCSS
    plt.plot(range(1, max_clusters + 1), wcss_values)
    plt.title('Within-Cluster Sum of Squares (WCSS)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

    return wcss_values

# This is the correct riemannian distance for the manifold the covariance matrices live on
def custom_distance(cov_matrix1, cov_matrix2):
    try:
        S = scipy.linalg.sqrtm(cov_matrix1)
    except Exception as e:
        print("Error computing square root of cov_matrix1:", e)
        return None

    try:
        S_inverse = scipy.linalg.inv(S)
    except Exception as e:
        print("Error computing inverse of S:", e)
        return None

    try:
        W = S_inverse @ cov_matrix2 @ S_inverse
    except Exception as e:
        print("Error computing W:", e)
        return None

    try:
        distance = np.sqrt(np.sum(np.log(np.linalg.eigvals(W))**2))
    except Exception as e:
        print("Error computing distance:", e)
        return None

    return distance

def frobenius_distance(matrix1, matrix2):
    """
    Compute the Frobenius norm distance between two matrices.

    Args:
    - matrix1 (ndarray): First matrix.
    - matrix2 (ndarray): Second matrix.

    Returns:
    - distance (float): Frobenius norm distance between the two matrices.
    """
    distance = np.linalg.norm(matrix1 - matrix2, 'fro')
    return distance

def correlation_distance(matrix1, matrix2):
    """
    Compute the correlation distance between two matrices.

    Args:
    - matrix1 (ndarray): First matrix.
    - matrix2 (ndarray): Second matrix.

    Returns:
    - distance (float): Correlation distance between the two matrices.
    """
    # Compute the correlation coefficients
    corr_coeff = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
    # Compute the correlation distance
    distance = 1 - corr_coeff
    return distance

def kl_divergence(matrix1, matrix2):
    """
    Compute the Kullback-Leibler (KL) divergence between two matrices.

    Args:
    - matrix1 (ndarray): First matrix.
    - matrix2 (ndarray): Second matrix.

    Returns:
    - distance (float): KL divergence between the two matrices.
    """
    # Flatten matrices and compute histograms
    hist1, _ = np.histogram(matrix1.flatten(), bins='auto', density=True)
    hist2, _ = np.histogram(matrix2.flatten(), bins='auto', density=True)
    # Remove zeros to avoid division by zero
    hist1 = hist1[hist1 != 0]
    hist2 = hist2[hist2 != 0]
    # Compute KL divergence
    distance = np.sum(hist1 * np.log(hist1 / hist2))
    return distance


def plot_hierarchical_clustering(all_covariance_matrices, custom_distance, color_ranges):
    # Compute pairwise distances and distance matrix
    num_matrices = len(all_covariance_matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))

    for i in range(num_matrices):
        for j in range(i, num_matrices):
            cov_matrix1 = all_covariance_matrices[i]
            cov_matrix2 = all_covariance_matrices[j]
            distance = custom_distance(cov_matrix1, cov_matrix2)

            # Store the distance in the distance matrix
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance 

    # Define labels for the dendrogram
    labels = np.arange(1, len(distance_matrix) + 1)

    # Define a function to format labels with colors
    def format_label(label):
        for start, end in color_ranges:
            if start <= label <= end:
                return f'\033[38;5;{31 + (label % 6)}m{label}\033[0m'  # Modify label color
        return label

    # Apply the formatting function to all labels
    formatted_labels = [format_label(label) for label in labels]

    # Compute hierarchical clustering
    Z = linkage(distance_matrix, method='average')  # Adjust the method as needed

    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z, color_threshold=6, labels=formatted_labels)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Worm (=specific cov matrix)')
    plt.ylabel('Distance')
    plt.show()



'''
from sklearn.cluster import KMeans

# Compute pairwise distances and distance matrix
num_matrices = len(all_covariance_matrices)
distance_matrix = np.zeros((num_matrices, num_matrices))

for i in range(num_matrices):
    for j in range(i, num_matrices):
        cov_matrix1 = all_covariance_matrices[i]
        cov_matrix2 = all_covariance_matrices[j]
        distance = custom_distance(cov_matrix1, cov_matrix2)

        # Store the distance in the distance matrix
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance 

print(distance_matrix)

# kmeans clustering with custom distance metric
def kmeans_with_distance_matrix_plot(distance_matrix, n_clusters=5, random_state=None):
    # Create a KMeans object
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    
    # Fit KMeans to the precomputed distance matrix
    kmeans.fit(distance_matrix)
    
    # Get cluster labels
    cluster_labels = kmeans.labels_
    print(cluster_labels)
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for cluster_label in range(n_clusters):
        indices = np.where(cluster_labels == cluster_label)[0]
        plt.scatter(indices, [cluster_label] * len(indices), label=f'Cluster {cluster_label}', color=f'C{cluster_label}', alpha=0.6)
    
    plt.xlabel('Covariance Matrix Index')
    plt.ylabel('Cluster Label')
    plt.title('Euclidean Cluster Assignment of Covariance Matrices')
    plt.legend()
    plt.grid(True)
    plt.show()


kmeans_with_distance_matrix_plot(distance_matrix, n_clusters=5, random_state=42)

'''


