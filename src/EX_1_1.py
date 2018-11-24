from sklearn import model_selection
from sklearn.mixture import GaussianMixture
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np

def runEx_1_1(X, attributeNames): 
    N, M = X.shape

    # Range of K's to try
    KRange = range(1,4)
    T = len(KRange)

    covar_type = 'full'     # you can try out 'diag' as well
    reps = 3                # number of fits with different initalizations, best result will be kept

    for z in range(0,10): 
        # Allocate variables
        BIC = np.zeros((T,))
        AIC = np.zeros((T,))
        CVE = np.zeros((T,))

        # K-fold crossvalidation
        CV = model_selection.KFold(n_splits=10,shuffle=True)

        for t,K in enumerate(KRange):
                #print('Fitting model for K={0}'.format(K))
                # Fit Gaussian mixture model
                gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)
                # Get BIC and AIC
                BIC[t,] = gmm.bic(X)
                AIC[t,] = gmm.aic(X)

                # For each crossvalidation fold
                for train_index, test_index in CV.split(X):
        
                    # extract training and test set for current CV fold
                    X_train = X[train_index]
                    X_test = X[test_index]

                    # Fit Gaussian mixture model to X_train
                    gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)
                    # compute negative log likelihood of X_test
                    CVE[t] += gmm.score_samples(X_test).sum() * -1
        figure(z+1)
        plot(KRange, BIC,'-*b')
        plot(KRange, AIC,'-xr')
        plot(KRange, 2*CVE,'-ok')
        legend(['BIC', 'AIC', 'Crossvalidation'])
        xlabel('K')
    gmm = GaussianMixture(n_components=3, covariance_type=covar_type, n_init=reps).fit(X)
    print(attributeNames)
    print(gmm.means_)