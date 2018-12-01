# exercise 11.3.1
#
# Author Duran Köse S147153 (70%)
# Taras Karpin s153067(20%)
# Janus Bastian Lansner S145349(10%)
# 
import numpy as np
from matplotlib.pyplot import figure, bar, title, show, plot, xticks
from scipy.stats.kde import gaussian_kde
import sys
sys.path.append('./Tools')
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

def runEx_2_1(data): 
    X = data
    N, M = X.shape
    
    widths = 2.0**np.arange(-10,10)
    logP = np.zeros(np.size(widths))
    for i,w in enumerate(widths):
        f, log_f = gausKernelDensity(X, w)
        logP[i] = log_f.sum()
    val = logP.max()
    ind = logP.argmax()

    width=widths[ind]
    print('Optimal estimated width is: {0}'.format(width))

    # Estimate density for each observation not including the observation
    # itself in the density estimate
    density, log_density = gausKernelDensity(X, width)

    # Sort the densities
    i = (density.argsort(axis=0)).ravel()
    density = density[i]

    # Display the index of the lowest density data object
    print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))
    # Plot density estimate of outlier score
    figure(2101)
    plotPoints = 30
    bar(range(plotPoints),density[:plotPoints].reshape(-1,))
    xticks(range(plotPoints), i[:plotPoints])
    title('Density estimate')

    ### K-neighbors density estimator
    # Neighbor to use:
    K = 5

    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)

    density_glob = 1./(D.sum(axis=1)/K)

    # Sort the scores
    i = density_glob.argsort()
    density = density_glob[i]

    # Plot k-neighbor estimate of outlier score (distances)
    figure(2102)
    bar(range(plotPoints),density[:plotPoints])
    xticks(range(plotPoints), i[:plotPoints])
    title('KNN density: Outlier score')

    ### K-nearest neigbor average relative density
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)
    density = 1./(D.sum(axis=1)/K)
    avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

    # Sort the avg.rel.densities
    i_avg_rel = avg_rel_density.argsort()
    avg_rel_density = avg_rel_density[i_avg_rel]

    # Plot k-neighbor estimate of outlier score (distances)
    figure(2103)
    bar(range(plotPoints),avg_rel_density[:plotPoints])
    xticks(range(plotPoints), i_avg_rel[:plotPoints])
    title('KNN average relative density: Outlier score')

