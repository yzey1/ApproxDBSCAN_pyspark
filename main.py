from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from graphframes import GraphFrame
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.appName("ApproxDBSCAN").getOrCreate()

from sklearn import datasets
from partitioning import parallelize_data
from approxDBSCAN import ApproxDBSCAN

from evaluate import *
from plot import plot_clustering_points

if __name__ == "__main__":

    # set parameters
    eps = 0.5
    min_pts = 5
    n_pa_each_dim = [2 for _ in range()]
    
    # read data
    iris = datasets.load_iris()
    data = iris.data
    
    # parallelize data
    partitioned_data, n_grid_each_dim = parallelize_data(data, eps, min_pts, n_pa_each_dim, sc)
    
    # run approxDBSCAN
    ad_cluster = ApproxDBSCAN(partitioned_data, eps, min_pts, n_grid_each_dim, n_pa_each_dim)
    print(ad_cluster)
    
    # get clustering result
    iris_result = get_point_cluster_df(data, ad_cluster)
    iris_result['sklearn_cluster'] = sklearn_dbscan(data, eps, min_pts)

    metrics = evaluate(data, iris_result['cluster_id'], iris_result['sklearn_cluster'])

    plot_clustering_points(data, iris_result, 'IRIS_ApproxDBSCAN')