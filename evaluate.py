import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.cluster import DBSCAN
from utils import reindex_id

def sklearn_dbscan(X, eps, min_pts):
    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    dbscan.fit(X)
    return dbscan.labels_


def get_point_cluster_df(X, adbscan):
    result = pd.DataFrame(list(range(len(X))), columns=["point_id"])
    result = result.merge(adbscan, on="point_id", how="left")
    result = result.fillna(-1)
    result["cluster_id"] = reindex_id(result["cluster_id"])
    return result

# clustering evaluation metrics
def evaluate(X, y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    sc = silhouette_score(X, y_pred)
    
    print('ARI: {:.4f}'.format(ari))
    print('AMI: {:.4f}'.format(ami))
    print('Silhouette Coefficient: {:.4f}'.format(sc))
    
    return ari, ami, sc