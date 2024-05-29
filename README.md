# ApproxDBSCAN_pyspark

This is a parallel implementation of [Approximate DBSCAN](https://sites.google.com/view/approxdbscan) algorithm on PySpark.

To understand the detailed implementation:
- see the output of each step in `experiment.ipynb`.

To view use cases for the algorithm
- refer to `main.ipynb`
- e.g., ![egplot](figs/random_ApproxDBSCAN.png)

`partitioning.py` : Space partitioning operations. Used to create grids and to create partitioned spaces for parallel computing partitions.

`approxDBSCAN.py` : Implementation of the Parallel Approximate DBSCAN Algorithm.

`evaluate.py` : Evaluate the clustering result.

`plot.py`: Visualize the clustering result.