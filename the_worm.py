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

directory = L1
directory_name = "L1"

cov_matrices = []
# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(os.path.join(directory, filename)):
        # Calculate covariance matrix for the file
        file_path = os.path.join(directory, filename)
        covariance_matrix = calculate_covariance_matrix(file_path)
        cov_matrices.append(covariance_matrix)
        
    # Somehow it doesn't let me read the other ones. 
    if filename == '102122_N2_adult_crawl_0001.txt':
        break


pairwise_distances = []

# Determine the number of covariance matrices
num_matrices = len(cov_matrices)

# Initialize a 2D array to store pairwise distances
distance_matrix = np.zeros((num_matrices, num_matrices))

# Iterate over each pair of covariance matrices
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

        # For the histogram
        pairwise_distances.append(distance)

# This snippet of code just plots a single row of the distance matrix, don't really know whether that is useful.
'''
pairwise_distances = np.array(pairwise_distances)
worm_label = 2
x_labels = np.arange(len(distance_matrix[worm_label]))
plt.bar(x_labels, distance_matrix[worm_label])
plt.xticks(x_labels)
plt.title("Distance between worm #" + str(worm_label) + " and the others" + str(directory_name))
plt.xlabel("worm #")
plt.ylabel("distance")
plt.show()
'''

# Print or further process the distance matrix as needed
print(distance_matrix)
plt.imshow(distance_matrix)
plt.colorbar()
plt.title('pairwise cov matrix distance between' + str(directory_name) + 'worms')
plt.xlabel("worm #")
plt.ylabel("worm #")
plt.show()

