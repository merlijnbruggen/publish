import os
import csv
import numpy as np
import scipy.linalg
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

directory_list = [L1, L1_Late, L2, L3, L4, Adult]
# First direcotry to compare
directory1 = L1
directory_name1 = "L1"

# Second directory to compare
directory2 = L1_Late
directory_name2 = "L1_Late"

# Compute the distance between worms within one developmental stage
# It needs a list of covariance matrices, they need to be calculated separately. 
def compute_pairwise_distances(cov_matrices):  
    pairwise_distances = []
    num_matrices = len(cov_matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))

    for i in range(num_matrices):
        for j in range(i, num_matrices):  # Here the i is warranted because that takes advantage of the symmetry of the distance matrix
            cov_matrix1 = cov_matrices[i]
            cov_matrix2 = cov_matrices[j]

            # Compute the distance between each pair of covariance matrices
            S = scipy.linalg.sqrtm(cov_matrix1)
            S_inverse = scipy.linalg.inv(S)
            W = S_inverse @ cov_matrix2 @ S_inverse
            distance = np.sqrt(np.sum(np.log(np.linalg.eigvals(W))**2))

            # Store the distance in the distance matrix
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Distance matrix is symmetric
        
    normalization = np.max(distance_matrix)
    #distance_matrix = distance_matrix/normalization
    
    '''
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.title('pairwise cov matrix distance between ' + str(directory_name1) + ' worms (normalized)')
    plt.xlabel('worm #; max distance:' + str(normalization))
    plt.ylabel("worm #")
    plt.show()
    print(normalization)
    '''

    return distance_matrix

'''
cov_matrices1 = []  # Directory 1
for filename in os.listdir(directory1):
    if os.path.isfile(os.path.join(directory1, filename)):
        file_path = os.path.join(directory1, filename)
        covariance_matrix = calculate_covariance_matrix(file_path)
        cov_matrices1.append(covariance_matrix)
    if filename == '102122_N2_adult_crawl_0001.txt':
        break
compute_pairwise_distances(cov_matrices1)
'''


# Compute the distance between two developmental stages 
def compute_distance_matrix(directory1, directory2, calculate_covariance_matrix):
    
    cov_matrices1 = []  # Directory 1
    for filename in os.listdir(directory1):
        if os.path.isfile(os.path.join(directory1, filename)):
            file_path = os.path.join(directory1, filename)
            covariance_matrix = calculate_covariance_matrix(file_path)
            cov_matrices1.append(covariance_matrix)
        if filename == '102122_N2_adult_crawl_0001.txt':
            break
    cluster_distance1 = compute_pairwise_distances(cov_matrices1)
   
    cov_matrices2 = []  # Directory 2
    for filename in os.listdir(directory2):
        if os.path.isfile(os.path.join(directory2, filename)):
            file_path = os.path.join(directory2, filename)
            covariance_matrix = calculate_covariance_matrix(file_path)
            cov_matrices2.append(covariance_matrix)
        if filename == '102122_N2_adult_crawl_0001.txt':
            break
    cluster_distance2 = compute_pairwise_distances(cov_matrices2)

    num_matrices = min(len(cov_matrices1), len(cov_matrices2))
    distance_matrix = np.zeros((num_matrices, num_matrices))

    for i in range(num_matrices):
        for j in range(num_matrices):    #This was range(i+1, numn_matrices) but that is not correct
            cov_matrix1 = cov_matrices1[i]
            cov_matrix2 = cov_matrices2[j]

            S = scipy.linalg.sqrtm(cov_matrix1)
            S_inverse = scipy.linalg.inv(S)
            W = S_inverse @ cov_matrix2 @ S_inverse
            distance = np.sqrt(np.sum(np.log(np.linalg.eigvals(W))**2))

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance


    # i (rows) are the directory 1 worms
    # j (columns) are the directory 2 worms

    '''
    # Normalize the distance matrix
    normalization = max(np.max(cluster_distance1), np.max(cluster_distance2))
    print('normalization is:' + str(normalization))
    distance_matrix = distance_matrix/normalization
    '''
    
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.title('pairwise cov matrix distance between ' + str(directory_name1) + ' and ' + str(directory_name2))
    plt.xlabel("worm # (" + str(directory_name2) + ")")
    plt.ylabel("worm # (" + str(directory_name1) + ")")
    plt.show()

    return distance_matrix


#compute_distance_matrix(directory1, directory2, calculate_covariance_matrix)
 

def compute_within_cluster_distances(directory, calculate_covariance_matrix):
    cov_matrices = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_path = os.path.join(directory, filename)
            covariance_matrix = calculate_covariance_matrix(file_path)
            cov_matrices.append(covariance_matrix)
        if filename == '102122_N2_adult_crawl_0001.txt':
            break
    
    pairwise_distances = []
    num_matrices = len(cov_matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))

    for i in range(num_matrices):
        for j in range(i + 1, num_matrices):
            cov_matrix1 = cov_matrices[i]
            cov_matrix2 = cov_matrices[j]

            # Compute the distance between each pair of covariance matrices
            S = scipy.linalg.sqrtm(cov_matrix1)
            S_inverse = scipy.linalg.inv(S)
            W = S_inverse @ cov_matrix2 @ S_inverse
            distance = np.sqrt(np.sum(np.log(np.linalg.eigvals(W))**2))

            # Store the distance in the distance matrix
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Distance matrix is symmetric

            # Append the distance to the list of pairwise distances
            pairwise_distances.append(distance)

    return distance_matrix, pairwise_distances


'''
# Plot the mean distance in each cluster as a histogram. 
mean_distances = []
directory_list_string = ['L1', 'L1_Late', 'L2', 'L3', 'L4', 'Adult']

for cluster in directory_list:
    distance_matrix, pairwise_distances = compute_within_cluster_distances(cluster, calculate_covariance_matrix)
    mean_distance = np.mean(pairwise_distances)
    mean_distances.append(mean_distance)
    print(mean_distances)

plt.bar(directory_list_string, mean_distances)
plt.xlabel('Cluster')
plt.ylabel('Mean Distance')
plt.title('Mean Distance within Each Cluster')
plt.show()
'''

def calculate_difference_pca(directory1, directory2, calculate_covariance_matrix):
    # List all files in the directories
    file_list1 = os.listdir(directory1)
    file_list2 = os.listdir(directory2)

    # Assuming you want to use the first file in each directory
    file_path1 = os.path.join(directory1, file_list1[0])
    file_path2 = os.path.join(directory2, file_list2[0])

    # Compute the covariance matrices for each directory
    cov_matrix_directory1 = calculate_covariance_matrix(file_path1)
    cov_matrix_directory2 = calculate_covariance_matrix(file_path2)

    # Compute the difference between the two covariance matrices and see what its principal components are
    difference_matrix = (cov_matrix_directory1 - cov_matrix_directory2) / 10

    w, v = np.linalg.eigh(difference_matrix)  # find eigenvalues and eigenvectors of a real symmetric matrix

    sort = np.argsort(w)[::-1]  # sort eigenvalues from largest to smallest
    w = w[sort]
    v = v[:, sort]  # sort eigenvectors accordingly

    plt.figure()
    plt.plot(w)
    plt.xlabel('eigenmode')
    plt.ylabel('eigenvalue')
    plt.show()

    w = w / np.sum(w)  # normalized eigenvalues = variance along each principal component
    wsum = np.cumsum(w)  # cumulative sum of normalized eigenvalues = variance captured

    plt.figure(figsize=(4, 4))
    plt.plot(np.arange(1, 9), wsum[:8], 'o-')
    plt.ylim(0, 1.5)
    plt.xticks(np.arange(1, 9))
    plt.xlabel('# principal components', fontsize=16)
    plt.ylabel('cumulative variance', fontsize=16)
    plt.show()

    K = 4  # number of principal components to use
    PC = v[:, :K]  # eigenvectors for these PCs; each column is one PC

    fig, ax = plt.subplots(1, K, figsize=(5 * K, 4))

    for i in range(K):
        ax[i].plot(PC[:, i])
        ax[i].set_title(f'PC{i + 1}')
        ax[i].set_xlabel('worm segment')
        ax[i].set_ylabel('angle (rad)')
    fig.suptitle('PCs of difference ' + str(directory_name1) + 'and ' + str(directory_name2))
    plt.show()

    D = len(cov_matrix_directory1[:, 0])

    def angle2pos(angles):
        """
        convert from angles of consecutive segments of unit length to positions of their endpoints.
        inputs:
        angles: 1-d array, angles of consecutive segments.
        outputs:
        pos: 2-d array, each row is (x,y) coordinates of an endpoint, centered at zeros.
        """
        dx = np.cos(angles)
        dy = np.sin(angles)
        xsum = np.cumsum(dx)
        ysum = np.cumsum(dy)
        pos = np.zeros((D + 1, 2))
        pos[1:, 0] = xsum
        pos[1:, 1] = ysum
        mean = np.mean(pos, axis=0)
        pos = pos - mean
        return pos

    pos_all = []
    for i in range(K):
        pos = angle2pos(PC[:, i] * 5)  # convert PC (with amplitude 5) to positions
        pos_all.append(pos)

    fig, ax = plt.subplots(1, K, figsize=(2 * K, 4))
    for i in range(K):
        pos = pos_all[i]
        ax[i].plot(pos[:, 0], pos[:, 1], '.-')
        ax[i].set_aspect('equal')
        ax[i].axis('off')
    fig.suptitle('Eigenworms of difference ' + str(directory_name1) + 'and ' + str(directory_name2))
    plt.show()

def calculate_pca(directory, calculate_covariance_matrix):
    # List all files in the directory
    file_list = os.listdir(directory)

    # Only use first file
    file_path = os.path.join(directory1, file_list[0])
    
    # Calculate covariance matrix
    cov_matrix_directory = calculate_covariance_matrix(file_path)
    

    w, v = np.linalg.eigh(cov_matrix_directory)  # find eigenvalues and eigenvectors of a real symmetric matrix

    sort = np.argsort(w)[::-1]  # sort eigenvalues from largest to smallest
    w = w[sort]
    v = v[:, sort]  # sort eigenvectors accordingly

    plt.figure()
    plt.plot(w)
    plt.xlabel('eigenmode')
    plt.ylabel('eigenvalue')
    plt.show()

    w = w / np.sum(w)  # normalized eigenvalues = variance along each principal component
    wsum = np.cumsum(w)  # cumulative sum of normalized eigenvalues = variance captured

    plt.figure(figsize=(4, 4))
    plt.plot(np.arange(1, 9), wsum[:8], 'o-')
    plt.ylim(0, 1.5)
    plt.xticks(np.arange(1, 9))
    plt.xlabel('# principal components', fontsize=16)
    plt.ylabel('cumulative variance', fontsize=16)
    plt.show()

    K = 4  # number of principal components to use
    PC = v[:, :K]  # eigenvectors for these PCs; each column is one PC

    fig, ax = plt.subplots(1, K, figsize=(5 * K, 4))

    for i in range(K):
        ax[i].plot(PC[:, i])
        ax[i].set_title(f'PC{i + 1}')
        ax[i].set_xlabel('worm segment')
        ax[i].set_ylabel('angle (rad)')
    fig.suptitle('PCs of ' + str(directory_name1))
    plt.show()

    D = len(cov_matrix_directory[:, 0])

    def angle2pos(angles):
        """
        convert from angles of consecutive segments of unit length to positions of their endpoints.
        inputs:
        angles: 1-d array, angles of consecutive segments.
        outputs:
        pos: 2-d array, each row is (x,y) coordinates of an endpoint, centered at zeros.
        """
        dx = np.cos(angles)
        dy = np.sin(angles)
        xsum = np.cumsum(dx)
        ysum = np.cumsum(dy)
        pos = np.zeros((D + 1, 2))
        pos[1:, 0] = xsum
        pos[1:, 1] = ysum
        mean = np.mean(pos, axis=0)
        pos = pos - mean
        return pos

    pos_all = []
    for i in range(K):
        pos = angle2pos(PC[:, i] * 5)  # convert PC (with amplitude 5) to positions
        pos_all.append(pos)
    '''
    fig, ax = plt.subplots(1, K, figsize=(2 * K, 4))
    for i in range(K):
        pos = pos_all[i]
        ax[i].plot(pos[:, 0], pos[:, 1], '.-')
        ax[i].set_aspect('equal')
        ax[i].axis('off')
    fig.suptitle('Eigenworms of ' + str(directory_name))
    plt.show()
    '''




