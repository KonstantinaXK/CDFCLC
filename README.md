# tetrad_test.py
In this file we implement a confirmatory tetrad test, experiment with different test statistics and compare their performance. We run the test for different inputs, that is, a variety of observation matrices, with different sizes, sampled both at random and accoding to our linear latent variable model. Moreover, we plot the results using histograms and scatter plots.

# kl_div.py
In this file we minimize the Kullback–Leibler (KL) divergence between two multivariate normal distributions where one has a given covariance matrix S and the other has an estimated covariance matrix Σ_0, necessary for the bootstrap step of the tetrad test. 

# chi2.py
In this file we compare the distribution of the (original) tetrad test (using test statistic T_0) to the chi2-distribution.
We calculate and visualize their distance as the size of the observation matrix get increased.

# FOFC.py
In this file we implement the FindOneFactorClusters algorithm that receives as input an observation matrix and gives a list of non-intersecting clusters of variables, where it is implied that the variables within the same cluster have a common latent variable. We test it on simulated data and we compare the obtained and the true DAGs.
