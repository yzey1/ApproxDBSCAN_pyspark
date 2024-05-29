from utils import *
from graphframes import GraphFrame

# find core points in each partition
def find_core_points(pa, eps, min_pts):
    pa = list(pa)
    for p in pa:
        pid = p[0]
        grids = p[1]
        
        points = []
        for grid in grids:
            for p in grid[1]:
                points.append([pid, grid[0], p])
                
        for paid, gid, (pid,value) in points:
            count = 0
            core_flag = 0
            for _, _, (_,value2) in points:
                if value == value2:
                    continue
                if np.linalg.norm(np.array(value) - np.array(value2)) <= eps:
                    count += 1
                    if count >= min_pts:
                        core_flag = 1
                        break
            yield ((paid, gid), (pid, value), core_flag)

def find_core_cells(pa):
    pa = list(pa)
    core_cells = {}
    for key, point, flag in pa:
        if flag == 1:
            if key not in core_cells:
                core_cells[key] = []
            core_cells[key].append(point)
    for key in core_cells:
        yield ((key[0], key[1]), core_cells[key])
    
def get_distance_matrix(points1, points2):
    distance_matrix = np.zeros((len(points1), len(points2)))
    for i in range(len(points1)):
        for j in range(len(points2)):
            distance_matrix[i][j] = np.linalg.norm(np.array(points1[i]) - np.array(points2[j]))
    return distance_matrix

def find_neighbor_cell(pa, eps, rho):
    pa = list(pa)
    partitions = {}
    
    for (paid, gid), points in pa:
        if paid not in partitions:
            partitions[paid] = []
        partitions[paid].append((gid, points))
    
    for pid, grids in partitions.items():
        for i, (gid, points) in enumerate(grids):
            point_values = [p[1] for p in points]
            for gid2, points2 in grids[i+1:]:
                point_values2 = [p[1] for p in points2]
                distance_matrix = get_distance_matrix(point_values, point_values2)
                isNeighbor = np.sum(distance_matrix <= eps)
                if isNeighbor >= 1:
                    yield (pid, gid, gid2)
                else:
                    isApproxNeighbor = np.sum(distance_matrix <= eps*(1+rho))
                    if isApproxNeighbor >= 1:
                        if generate_random_binary():
                            yield (pid, gid, gid2)

def compute_connected_components(core_cells, neighbor_pairs, pos_to_gid, pos_to_paid, n_pa_each_dim, sc):
    """
    Compute the connected components of a graph using Apache Spark's GraphFrame.

    Args:
        core_cells (RDD): RDD containing the core cells.
        neighbor_pairs (RDD): RDD containing the neighbor pairs.
        pos_to_gid (dict): Dictionary mapping position to global ID.
        pos_to_paid (dict): Dictionary mapping position to partition ID.
        n_pa_each_dim (list): List containing the number of partitions in each dimension.
        sc (SparkContext): SparkContext object.

    Returns:
        RDD: RDD containing the connected components.

    Raises:
        None
    """

    # create vertices and edges
    v = core_cells.map(lambda x: (pos_to_paid[x[0][0]], pos_to_gid[x[0][1]],)).toDF(["paid", "id"])
    v = v.repartition(int(np.prod(n_pa_each_dim)), "paid")
    e = neighbor_pairs.map(lambda x: (pos_to_paid[x[0]], pos_to_gid[x[1]], pos_to_gid[x[2]])).toDF(["paid", "src", "dst"])
    e = e.repartition(int(np.prod(n_pa_each_dim)), "paid")

    # create graph
    g = GraphFrame(v, e)
    
    # set Checkpoint directory
    sc.setCheckpointDir("checkpoints")

    # compute connected components
    # not optimal to use connectedComponents(), since it will shuffle the data and lose the partition.
    # TODO: New implementation required.
    connectedComponent = g.connectedComponents()
    
    # remove duplicates
    connectedComponent = connectedComponent.dropDuplicates(['id'])
    
    return connectedComponent.rdd

def merge_partitioned_data(points_with_flag, pos_to_gid):
    """
    Merge partitioned data by removing duplicates and updating flags.

    Args:
        points_with_flag (RDD): RDD containing points with flags.
        pos_to_gid (dict): Dictionary mapping position to group ID.

    Returns:
        RDD: RDD containing merged points with updated flags.
    """
    
    def _remove_duplicates(x):
        """
        Remove duplicates within a group and update flags.

        Args:
            x (tuple): Tuple containing group ID and list of points.

        Yields:
            tuple: Tuple containing group ID and point with updated flag.
        """
        gid = x[0]
        points = x[1]
        points_flag = {}
        for p in points:
            if p[0] not in points_flag:
                points_flag[p[0]] = p[1]
            else:
                if (p[1] == 1) & (points_flag[p[0]] == 0):
                    points_flag[p[0]] = 1
        for k, v in points_flag.items():
            yield (gid, (k, v))

    points_flagged = points_with_flag.map(lambda x: (pos_to_gid[x[0][1]], (x[1][0], x[2])))
    points_flagged = points_flagged.groupByKey().mapValues(list)
    points_flagged = points_flagged.flatMap(lambda x: _remove_duplicates(x))
    
    return points_flagged

def ApproxDBSCAN(partitioned_rdd, eps, min_pts, rho, n_grid_each_dim, n_pa_each_dim, sc):
    """
    Perform Approximate DBSCAN clustering algorithm on a partitioned RDD.

    Parameters:
    - partitioned_rdd (RDD): The partitioned RDD containing the data points.
    - eps (float): The maximum distance between two points to be considered neighbors.
    - min_pts (int): The minimum number of points required to form a dense region.
    - rho (float): The density threshold for identifying core cells.
    - n_grid_each_dim (int): The number of grid cells in each dimension for grid indexing.
    - n_pa_each_dim (int): The number of partition cells in each dimension for partitioning.
    - sc (SparkContext): The SparkContext object.

    Returns:
    - cluster_df (pandas DataFrame): The resulting clusters as a pandas DataFrame.
    """

    # Step 1: Compute the core points
    points_with_flag = partitioned_rdd.mapPartitions(lambda x: find_core_points(x, eps, min_pts))
    
    # Step 2: Compute the core cells
    core_cells = points_with_flag.mapPartitions(lambda x: find_core_cells(x))
    
    # Step 3: Find eps-neighbor cell pairs
    neighbor_pairs = core_cells.mapPartitions(lambda x: find_neighbor_cell(x, eps, rho))
    
    # Step 4: Create graph and find connected components
    pos_to_gid, _ = grid_index_mapping(n_grid_each_dim)
    pos_to_paid, _ = grid_index_mapping(n_pa_each_dim)
    connectedComponent = compute_connected_components(core_cells, neighbor_pairs, pos_to_gid, pos_to_paid, n_pa_each_dim, sc)
    
    # Step 5: Merge partitioned data
    points_flagged = merge_partitioned_data(points_with_flag, pos_to_gid)
    
    # Step 6: Assign points to clusters
    cluster = connectedComponent.map(lambda x: (x[1], x[2]))\
                                .join(points_flagged)\
                                .map(lambda x: (x[1][0], x[1][1][0], x[1][1][1]))
    
    # Convert to pandas dataframe and output
    cluster_df = cluster.toDF(["cluster_id", "point_id", "core_flag"])
    cluster_df = cluster_df.toPandas()
    
    return cluster_df
    