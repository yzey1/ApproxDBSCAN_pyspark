import numpy as np
from itertools import product

def cal_grid_side_len(eps, n_features):
    return eps/np.sqrt(n_features)

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
    n_features = min_max_bounds.shape[0]
    grid_side_len = cal_grid_side_len(eps, n_features)
    gridnum_each_dim = []
    for i in range(n_features):
        gridnum_each_dim.append(int((min_max_bounds[i][1] - min_max_bounds[i][0]) / grid_side_len))
    return gridnum_each_dim


# transform the tuple index (based on position) to integer index
def grid_index_mapping(n_grid_each_dim):
    grid_index = []
    for i in n_grid_each_dim:
        grid_index.append(range(i))
    grid_index = list(product(*grid_index))
    pos_to_gid = {}
    gid_to_pos = {}
    for n in range(len(grid_index)):
        pos_to_gid[grid_index[n]] = n
        gid_to_pos[n] = grid_index[n]
    return pos_to_gid, gid_to_pos

