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
    # PCA by computing SVD of Y

    f = figure(1251)
    title('NanoNose data: PCA')
    for c in range(C):
        # select indices belonging to class c:
        class_mask = NS_Y == c
        plot(Z[class_mask,i], Z[class_mask,j], 'o')
    legend(classNames)
    xlabel('PC{0}'.format(i+1))
    ylabel('PC{0}'.format(j+1)) 

    N, M = X.shape
    # Perform hierarchical/agglomerative clustering on data matrix
    Methods = ['single', 'average', 'complete']
    Metric = 'euclidean'
    i = 1
    for Method in Methods:
        print("printing")
        Z = linkage(X, method=Method, metric=Metric)

        # Compute and display clusters by thresholding the dendrogram
        Maxclust = 4
        cls = fcluster(Z, criterion='maxclust', t=Maxclust)
        figure(1200+i)
        i += 1
        clusterplot(X @ V, cls.reshape(cls.shape[0],1), y=y)

        # Display dendrogram
        max_display_levels=6
        figure(1200 + i,figsize=(10,4))
        i += 1
        dendrogram(Z, truncate_mode='level', p=max_display_levels)
