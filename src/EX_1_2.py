from matplotlib.pyplot import figure
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def runEx_1_2(X,y): 
    
    N, M = X.shape
    # Perform hierarchical/agglomerative clustering on data matrix
    Methods = ['single', 'average', 'complete']
    Metric = 'euclidean'
    i = 1
    for Method in Methods:
        print("printing")
        Z = linkage(X, method=Method, metric=Metric)

        # Compute and display clusters by thresholding the dendrogram
        Maxclust = 9
        cls = fcluster(Z, criterion='maxclust', t=Maxclust)
        figure(1200+i)
        i += 1
        clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

        # Display dendrogram
        max_display_levels=6
        figure(1200 + i,figsize=(10,4))
        i += 1
        dendrogram(Z, truncate_mode='level', p=max_display_levels)