import numpy as np
import scipy.linalg
import os
import csv
import math
import random 
import matplotlib.pyplot as plt

def calculate_covariance_matrix(file_path):
    # Open the .txt file in read mode
    with open(file_path, 'r', encoding='utf-8') as file:
        # Create a CSV reader object with tab delimiter
        reader = csv.reader(file, delimiter='\t')
        
        # Skip the first two rows
        next(reader)
        next(reader)
        
        # Initialize an empty list to store numerical data
        data = []
        
        # Iterate over the rows in the CSV reader
        for row in reader:
            # Attempt to convert each element in the row to a float
            numeric_row = [float(value) for value in row[1:11]]  # Assuming columns 1 to 10 contain numerical data
            
            # Append the numeric row to the data list
            data.append(numeric_row)
    
    # Convert the list of lists to a NumPy array
    data_array = np.array(data)
    
    # Calculate the covariance matrix
    x_mean = np.mean(data_array, axis=0)
    x_cent = data_array - x_mean    # centered data
    cov = np.dot(x_cent.T, x_cent) / data_array.shape[0]    # covariance matrix
    return cov
# Path to the directory containing your files
L1 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L1'
L1_Late = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_LateL1'
L2 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L2'
L3 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L3'
L4 = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_L4'
Adult = '/Users/merlijnbruggen/Desktop/The Worm and Markov Chains/doi_10_5061_dryad_stqjq2c8p__v20240130/dryad/Crawling/N2_Adult'

directory_list = [L1, L1_Late, L2, L3, L4]

# Load all covariance matrices
all_covariance_matrices = []
for directory in directory_list:
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_path = os.path.join(directory, filename)
            covariance_matrix = calculate_covariance_matrix(file_path)
            all_covariance_matrices.append(covariance_matrix)
        if filename == '102122_N2_adult_crawl_0001.txt':
            print("Error: reached file I couldn't read")
            break




# Compute the geodesic distance between 2 covariance matrices (=SPD matrices)
def geodesic_distance(cov_matrix1, cov_matrix2):
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

# When I run this function, the mean decreases. That is what I want 
def intrinsic_mean(cov_matrices):
    intrinsic_mean = []
    data = cov_matrices
    #TODO: I am 90% sure this caluclation is the correct algorithm. 
    eps = 5.0
    # Basis point (a matrix in this case) for the exponential and logarithm map on the manifold.
    M = np.identity(len(cov_matrices[0])) # This should be the identity matrix or some other SPD matrix
    
    while True:
        M_sqrt = np.abs(scipy.linalg.sqrtm(M))
        M_sqrt_inv = scipy.linalg.inv(M_sqrt)
        M2 = np.zeros((len(cov_matrices[0]),len(cov_matrices[0])))
        
        # Compute the Riemannian matrix logarithm 
        Y_m = np.zeros((len(cov_matrices[0]),len(cov_matrices[0])))
        for i in range(0, len(data)):
            Y_m += scipy.linalg.logm(M_sqrt_inv @ data[i] @ M_sqrt_inv)
        Y_m = 1/len(cov_matrices) * M_sqrt @ Y_m @ M_sqrt

    
        # Now the Riemannian exponential
        exponential_Y = M_sqrt @ scipy.linalg.expm(M_sqrt_inv @ Y_m @ M_sqrt_inv) @ M_sqrt
        
        # When ||Y_m|| gets below a certain threshold, stop the algorithm

        norm_Y_m = np.sqrt(np.matrix.trace(np.square(Y_m)))

        if norm_Y_m < eps:
            print("Threshold reached")
            break
        # Update the new basis point from M to the new one
        M = exponential_Y
        intrinsic_mean = M
        
        print(norm_Y_m)
    return intrinsic_mean

    
def assign_to_clusters(cov_matrices, centroids):
    cluster_labels = []
    for matrix in cov_matrices:
        min_distance = float('inf')
        min_cluster_idx = None
        for cluster_idx, centroid in enumerate(centroids):
            distance = geodesic_distance(matrix, centroid)
            if distance < min_distance:
                min_distance = distance
                min_cluster_idx = cluster_idx
        cluster_labels.append(min_cluster_idx)
    
    return cluster_labels


def Gclust(cov_matrices, num_clusters, max_iterations, show_plot):
    cluster_labels = np.array([None] * len(cov_matrices))  # Convert to NumPy array
    
    # Initialize centroids as random samples from cov_matrices
    centroids = random.sample(cov_matrices, num_clusters)
    
    count = 0
    assign_to_clusters(cov_matrices, centroids)

    # Repeat until convergence
    while True: 
        old_cluster_labels = cluster_labels.copy()  # Copy cluster labels
        
        # Assign each matrix to the nearest centroid
        for matrix_idx, matrix in enumerate(cov_matrices):
            min_distance = float('inf')
            min_centroid_idx = None

            for centroid_idx, centroid in enumerate(centroids):
                distance = geodesic_distance(matrix, centroid)
                
                if distance < min_distance:
                    min_distance = distance
                    min_centroid_idx = centroid_idx

            cluster_labels[matrix_idx] = min_centroid_idx

        # Compute new centroids as intrinsic means of clusters
        new_centroids = []
        for cluster_idx in range(num_clusters):
            cluster_matrices = [cov_matrices[i] for i, label in enumerate(cluster_labels) if label == cluster_idx]
            new_centroid = intrinsic_mean(cluster_matrices)
            new_centroids.append(new_centroid)
        
        # Check for convergence by comparing cluster labels
        if np.array_equal(old_cluster_labels, cluster_labels):
            print("Convergence reached")
            break  # Convergence reached
        
        # Update centroids
        centroids = new_centroids
        count += 1 
        
        if count >= max_iterations:
            print("Max iterations reached")
            break

        print("Iteration:", count)
    
    print("Cluster labels:", cluster_labels)
    print("Number of clusters:", num_clusters)

    # Plot the clusters
    if show_plot:
        plt.figure(figsize=(10, 6))
        for cluster_label in range(num_clusters):
            indices = np.where(cluster_labels == cluster_label)[0]
            plt.scatter(indices, [cluster_label] * len(indices), label=f'Cluster {cluster_label}', color=f'C{cluster_label}', alpha=0.6, marker='o', s=50)
        
        plt.xlabel('Covariance Matrix Index')
        plt.ylabel('Cluster Label')
        plt.title('Geodesic Cluster Assignment of Covariance Matrices')
        plt.legend()
        plt.grid(True)  # Adjust based on preference
        plt.show()

    return cluster_labels


intrinsic_mean(all_covariance_matrices)
#Gclust(all_covariance_matrices,num_clusters=5,max_iterations=10, show_plot=True)







# 1) Pick num_cluster data matrices as centroids
# 3) Compute geodesic distance of each datapoint to centroid
# 4) Assign datapoints to clusters based on this distance
# 5) Calculate the new intrinsic means of each cluster
# 6) Repeat 3-5 until convergence