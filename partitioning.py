from utils import *
from itertools import product
import numpy as np

def get_minmax_by_data(dataset):
    """
    Get the minimum and maximum bounds for each dimension in the dataset.

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
    for i in range(len(x)):
        # pos: the index of the bin in that dimension
        pos = np.digitize(x[i], bin_bounds[i]) - 1
        pos = min(partition_each_dim[i]-1, pos) # if the value is the max value, it should be in the last bin
        pos = max(0, pos) # if the value is the min value, it should be in the first bin
        pos_list.append(pos)
    return tuple(pos_list)


def find_buffer_location_id(x, bin_bounds, partition_each_dim, buffer_size):
    """
    Find the buffer location IDs for a given point in each dimension.

    Args:
        x (list): The coordinates of the point.
        bin_bounds (list): The boundaries of each bin in each dimension.
        partition_each_dim (list): The number of partitions in each dimension.
        buffer_size (int): The size of the buffer.

    Returns:
        list: A list of all possible combinations of buffer location IDs for the given point.

    """
    pos_list = []
    for i in range(len(x)):
        # pos: the index of the bin in that dimension
        pos = np.digitize(x[i], bin_bounds[i]) - 1
        pos = min(partition_each_dim[i]-1, pos) # if the value is the max value, it should be in the last bin
        pos = max(0, pos) # if the value is the min value, it should be in the first bin
        
        lower_bound = bin_bounds[i][pos]
        upper_bound = bin_bounds[i][pos+1]
        pos_buffer = [pos]
        
        if x[i] - lower_bound < buffer_size:
            for j in range(1, buffer_size+1):
                pos_buffer.append(max(0, pos-j))
        if upper_bound - x[i] < buffer_size:
            for j in range(1, buffer_size+1):
                pos_buffer.append(min(partition_each_dim[i]-1, pos+j))
        pos_list.append(pos_buffer)
        
    all_combinations = list(product(*pos_list))
    return all_combinations

def add_partition_id(x, grid_bins, n_pa_each_dim, buffer_size):
    buffer_loc_id = find_buffer_location_id(x[0], grid_bins, n_pa_each_dim, buffer_size)
    buffer_loc_id = list(set(buffer_loc_id))
    return [(blid, (x[0], x[1])) for blid in buffer_loc_id]

def get_partitioned_cells(rdd, grid_bins, n_pa_each_dim, n_features):
    # buffer size (number of cells to be included in half of the buffer region)
    buffer_size = int(np.ceil(np.sqrt(n_features)))
    rdd1 = rdd.flatMap(lambda x: add_partition_id(x, grid_bins, n_pa_each_dim, buffer_size))
    partitioned_rdd = rdd1.groupByKey().mapValues(list)
    # partitioned_rdd = rdd1.sortByKey()
    return partitioned_rdd


def parallelize_data(X, eps, n_pa_each_dim, sc):
    """
    Parallelizes the data and performs partitioning for ApproxDBSCAN algorithm.

    Args:
        X (numpy.ndarray): The input data array.
        eps (float): The maximum distance between two samples for them to be considered as neighbors.
        n_pa_each_dim (list): The number of partitions in each dimension.
        sc (pyspark.SparkContext): The Spark context.

    Returns:
        partitioned_rdd (pyspark.rdd.RDD): The partitioned RDD containing the data points grouped by partitions.
        n_grid_each_dim (list): The number of grids in each dimension.

    """
    # convert the data to list
    X = X.tolist()
    n_features = len(X[0])
    n_partitions = np.prod(n_pa_each_dim)
    
    # construct the grid
    min_max_bounds = get_minmax_by_data(X)
    n_grid_each_dim = cal_grid_num(min_max_bounds, eps)
    grid_bin_bounds = get_bin_bounds(min_max_bounds, n_grid_each_dim)

    # locate the points
    rdd = sc.parallelize(X, n_partitions)
    rdd = rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    grid_rdd = rdd.map(lambda x: (find_location_id(x[1], grid_bin_bounds, n_grid_each_dim), (x[0], x[1])))

    # group the points inside each grid
    grid_rdd = grid_rdd.groupByKey().mapValues(list)
    
    # divide the grid into partitions
    grid_min_max = np.array(([0 for _ in range(len(n_grid_each_dim))], [i-1 for i in n_grid_each_dim])).T
    grid_bins = get_bin_bounds(grid_min_max, n_pa_each_dim)

    # partition the cells
    partitioned_rdd = get_partitioned_cells(grid_rdd, grid_bins, n_pa_each_dim, n_features)
    
    return partitioned_rdd, n_grid_each_dim
