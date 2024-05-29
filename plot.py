# visualization of the clustering result
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_clustering_points(X, result, save_name='DBSCAN.pdf'):
    X = np.array(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=result["cluster_id"].astype(int))
    ax.set_title("Approximate DBSCAN")
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=result["sklearn_cluster"].astype(int))
    ax1.set_title("Traditional DBSCAN")
    plt.savefig(f"figs/{save_name}", dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.show()