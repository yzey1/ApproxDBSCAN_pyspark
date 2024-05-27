import numpy as np

# functions for partitioning data

def get_minmax_by_data(dataset):
    """
    Calculate the minimum and maximum bounds for each dimension in the dataset.

    Parameters:
    dataset (numpy.ndarray): The input dataset.

    Returns:
    numpy.ndarray: An array containing the minimum and maximum bounds for each dimension.
                   The shape of the array is (d, 2), where d is the number of dimensions in the dataset.
    """
    min_max_bounds = np.concatenate(([np.min(dataset, axis=0)], [np.max(dataset, axis=0)]), axis=0).T   # (d, 2)
    return min_max_bounds

def get_minmax_by_bins(bin_ind, bin_bounds):
    """
    Get the minimum and maximum bounds for each bin index.

    Parameters:
    - bin_ind (list): A list of bin indices.
    - bin_bounds (list): A list of bin bounds.

    Returns:
    - min_max_bounds (list): A list of tuples containing the minimum and maximum bounds for each bin index.
    """
    min_max_bounds = []
    for i in range(len(bin_ind)):
        min_max_bounds.append((bin_bounds[i][bin_ind[i]], bin_bounds[i][bin_ind[i]+1]))
    return min_max_bounds

# get the bound of each bin
def get_bin_bounds(min_max_bounds, partition_each_dim):
    """
    Calculate the bounds of each bin in each dimension.

    Args:
        min_max_bounds (list): A list of tuples representing the minimum and maximum bounds for each dimension.
        partition_each_dim (list): A list of integers representing the number of partitions in each dimension.

    Returns:
        list: A list of numpy arrays, where each array contains the bounds of the bins in that dimension.
    """
    bin_bounds = []
    for i in range(len(partition_each_dim)):
        Lower_bound = min_max_bounds[i][0]
        Upper_bound = min_max_bounds[i][-1]
        dim_bins = np.linspace(Lower_bound, Upper_bound, partition_each_dim[i]+1, endpoint=True)
        bin_bounds.append(dim_bins)
    return bin_bounds

# locate the point/cell in the partitioned space, return the index of space
def find_location_id(x, bin_bounds, partition_each_dim):
    """
    Finds the location ID for a given data point `x` based on the bin bounds and partition information.

    Args:
        x (list): The data point coordinates.
        bin_bounds (list): The bin boundaries for each dimension.
        partition_each_dim (list): The number of partitions in each dimension.

    Returns:
        tuple: The location ID of the data point.

    """
    pos_list = []
    for i in range(n_features):
        # pos: the index of the bin in that dimension
        pos = np.digitize(x[i], bin_bounds[i]) - 1
        pos = min(partition_each_dim[i]-1, pos) # if the value is the max value, it should be in the last bin
        pos = max(0, pos) # if the value is the min value, it should be in the first bin
        pos_list.append(pos)
    return tuple(pos_list)


# functions for constructing grids

# calculate grid number for each dimension in one partition
def cal_grid_num(min_max_bounds, eps):
    """
    Calculate the number of grids in each dimension based on the minimum and maximum bounds and the epsilon value.

    Args:
        min_max_bounds (list): A list of tuples representing the minimum and maximum bounds for each feature.
        eps (float): The epsilon value used for calculating the grid side length.

    Returns:
        list: A list containing the number of grids in each dimension.

    """
    global n_features
    grid_side_len = eps/np.sqrt(n_features)
    gridnum_each_dim = []
    for i in range(n_features):
        gridnum_each_dim.append(int((min_max_bounds[i][1] - min_max_bounds[i][0]) / grid_side_len))
    return gridnum_each_dim