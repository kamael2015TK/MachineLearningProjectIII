from matplotlib.pyplot import figure
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend

def runEx_1_2(X,y, NS_Y, C, classNames): 

    U,S,V = svd(X,full_matrices=False)
    V = V.T
    # Project the centered data onto principal component space
    Z = X @ V
    i = 0
    j = 1

    N, M = X.shape
    # Perform hierarchical/agglomerative clustering on data matrix
    Methods = ['single', 'average', 'complete']
    Metric = 'euclidean'
    i = 1
    for Method in Methods:
        print("printing")
        Z = linkage(X, method=Method, metric=Metric)

        # Compute and display clusters by thresholding the dendrogram
        Maxclust = 3
        cls = fcluster(Z, criterion='maxclust', t=Maxclust)
        figure(1200+i)
        title('PCA: hierarchical clustering using ' + Method + ' linkage function')
        i += 1
        clusterplot(X @ V, cls.reshape(cls.shape[0],1), y=NS_Y)

        # Display dendrogram
        max_display_levels=6
        figure(1200 + i,figsize=(10,4))
        i += 1
        dendrogram(Z, truncate_mode='level', p=max_display_levels)
        title('Hierarchical clustering using ' + Method + ' linkage function')
