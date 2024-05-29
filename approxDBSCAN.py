from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from graphframes import GraphFrame
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.appName("ApproxDBSCAN").getOrCreate()

from utils import *


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

def find_neighbor_cell(pa, eps):
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

def compute_connected_components(core_cells, neighbor_pairs, pos_to_gid, pos_to_paid, n_pa_each_dim):
    
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

def merge_partitioned_data(points_with_flag, X, pos_to_gid):
    
    def _remove_duplicates(x):
        gid = x[0]
        points = x[1]
        points_flag = {}
        for p in points:
            if p[0] not in points_flag:
                points_flag[p[0]] = p[1]
            else:
                if (p[1] == 1)&(points_flag[p[0]] == 0):
                    points_flag[p[0]] = 1
        for k, v in points_flag.items():
            yield (gid, (k, v))

    points_flagged = points_with_flag.map(lambda x: (pos_to_gid[x[0][1]], (x[1][0], x[2])))
    points_flagged = points_flagged.groupByKey().mapValues(list)
    points_flagged = points_flagged.flatMap(lambda x: _remove_duplicates(x))
    
    return points_flagged

def ApproxDBSCAN(X, partitioned_rdd, eps, min_pts, n_grid_each_dim, n_pa_each_dim):

    # Step 1: Compute the core points
    points_with_flag = partitioned_rdd.mapPartitions(lambda x: find_core_points(x, eps, min_pts))
    
    # Step 2: Compute the core cells
    core_cells = points_with_flag.mapPartitions(lambda x: find_core_cells(x))
    
    # Step 3: Find eps-neighbor cell pairs
    neighbor_pairs = core_cells.mapPartitions(lambda x: find_neighbor_cell(x, eps))
    
    # Step 4: Create graph and find connected components
    pos_to_gid, gid_to_pos = grid_index_mapping(n_grid_each_dim)
    pos_to_paid, paid_to_pos = grid_index_mapping(n_pa_each_dim)
    connectedComponent = compute_connected_components(core_cells, neighbor_pairs, pos_to_gid, pos_to_paid, n_pa_each_dim)
    
    # Step 5: Merge partitioned data
    points_flagged = merge_partitioned_data(points_with_flag, X, pos_to_gid)
    
    # Step 6: Assign points to clusters
    cluster = connectedComponent.map(lambda x: (x[1], x[2]))\
                                .join(points_flagged)\
                                .map(lambda x: (x[1][0], x[1][1][0], x[1][1][1]))
    
    # Convert to pandas dataframe and output
    cluster_df = cluster.toDF(["cluster_id", "point_id", "core_flag"])
    cluster_df = cluster_df.toPandas()
    
    return cluster_df
    