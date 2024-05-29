from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from graphframes import GraphFrame
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.appName("ApproxDBSCAN").getOrCreate()

from sklearn import datasets
from partitioning import parallelize_data
from approxDBSCAN import ApproxDBSCAN

if __name__ == "__main__":
    
    # set parameters
    eps = 0.5
    min_pts = 5
    n_pa_each_dim = [2 for _ in range()]
    
    # read data
    iris = datasets.load_iris()
    data = iris.data.tolist()
    partitioned_data, n_grid_each_dim = parallelize_data(data, 0.5, [2, 2, 2, 2])
    
    # run approxDBSCAN
    ad = ApproxDBSCAN(partitioned_data, eps, min_pts, n_grid_each_dim, n_pa_each_dim)
    print(ad)