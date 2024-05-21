import numpy as np

def spatial_split(dataset, partition_each_dim):
    
    # Get the bounds of the dataset
    bounds = np.concatenate(([np.min(dataset, axis=0)], [np.max(dataset, axis=0)]), axis=0).T 
    
    # Create the bins for each dimension
    bin_bounds = []
    for i in range(len(partition_each_dim)):
        # Lower_bound = bounds[i][0] - abs(bounds[i][0]) * 0.01
        # Upper_bound = bounds[i][-1] + abs(bounds[i][-1]) * 0.01
        dim_bins = np.linspace(*bounds[i], partition_each_dim[i]+1, endpoint=True)
        bin_bounds.append(dim_bins)
    
    # Index the data points
    indexed_data = []
    for id_pts in range(len(dataset)):
        pos_list = []
        for i in range(dataset.shape[1]):
            pos = np.digitize(dataset[id_pts][i], bin_bounds[i]) - 1
            pos = min(pos, partition_each_dim[i]-1)
            pos = max(pos, 0)
            pos_list.append(pos)
        indexed_data.append([tuple(pos_list), id_pts])
    
    # # Sort the data points
    # indexed_data.sort(key=lambda x: x[0])
    
    return indexed_data, bin_bounds

# Example
# from sklearn.datasets import make_blobs
# # Generate random dataset with 3 dimensions
# n_samples = 100  # Number of samples
# n_features = 3  # Number of dimensions
# centers = 3  # Number of clusters
# random_state = 42  # Random state for reproducibility
# X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
# indexed_data, bin_bounds = spatial_split(X, [2, 2, 2])
# print(indexed_data)
# print(bin_bounds)

# partition_each_dim = [2, 2]
# dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# indexed_data, bin_bounds = spatial_split(partition_each_dim, dataset)
# print(indexed_data)
# print(bin_bounds)

# Output
# [([0, 0], 0), ([1, 0], 1), ([0, 1], 2), ([1, 1], 3)]
# [array([0., 4., 8.]), array([0., 4., 8.])]
