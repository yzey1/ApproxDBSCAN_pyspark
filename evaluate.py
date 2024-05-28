import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

# clustering evaluation metrics
def evaluate(X, y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    sc = silhouette_score(X, y_pred)
    
    print('ARI: {:.4f}'.format(ari))
    print('AMI: {:.4f}'.format(ami))
    print('Silhouette Coefficient: {:.4f}'.format(sc))
    
    return ari, ami, sc