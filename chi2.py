# PLOTTING T_0 & PDF OF chi^2 (for different size of X)
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, chi2
from testing_algorithm import T_stat_0, T_stat_1


D = 4  # variables
iterations = 10000  # number of obs matrices
niter = np.array([10, 100, 1000, 2000, 3000, 4000, 5000])  # size of obs matrix X
np.random.seed(0)
T_0 = np.zeros([iterations, len(niter)])
for s in range(len(niter)):
    current_iter = niter[s]
    for t in range(iterations):  # 200 iterations for every niter=100, 1000, ...
        Z_0 = np.random.randn(current_iter)
        X = np.zeros((current_iter, D))  # observation matrix
        L = np.random.rand(D) * 2 - 1
        for d in range(D):
            epsilon = 1 - L[d] ** 2
            noise = np.random.randn(current_iter) * epsilon ** .5
            X[:, d] = L[d] * Z_0 + noise
        T_0[t, s] = T_stat_0(X)  #


# T_0 for different sizes of obs matrices X_i compared to chi^2
plt.figure(1)
plt.clf()
bins = np.linspace(0, 20, 31)
bins_cen = bins[1:] - np.diff(bins[:2]) / 2
h0, b0 = np.histogram(T_0[:, 0], density=True, bins=bins)
h1, b1 = np.histogram(T_0[:, 1], density=True, bins=bins)
h2, b2 = np.histogram(T_0[:, 2], density=True, bins=bins)

plt.plot(bins_cen, h0, label=f'$T_0$ (X:{niter[0]}x{D})')
plt.plot(bins_cen, h1, color='green', label=f'$T_0$ (X:{niter[1]}x{D})')
plt.plot(bins_cen, h2, color='red', label=f'$T_0$ (X:{niter[2]}x{D})')

chi2_pmf = np.diff(chi2.cdf(bins, df=2))
plt.plot(bins_cen, chi2_pmf, color='black', linewidth=2, label='$\chi^2$')
plt.title(f'T_stat_0 for different sizes of observation matrices X_i', fontsize=16)
plt.xlabel(f"tetrad test statistic", fontsize=14)
leg = plt.legend(loc='upper right')
dist = np.zeros(len(niter))
for i in range(len(niter)):
    h0, b0 = np.histogram(T_0[:, i], density=True, bins=bins)
    dist[i] = wasserstein_distance(bins_cen, bins_cen, u_weights=h0, v_weights=chi2_pmf)


# distance between distribution of T_0 and chi^2 as number of observations increased
plt.figure(2)
plt.plot(niter, dist, ".", ms=15)
plt.title(f'Distance between distribution of $T_0$ and $\chi^2$', fontsize=16)
plt.xlabel(f"number of observations", fontsize=14)
plt.ylabel("Wasserstein distance", fontsize=14)
plt.show()
plt.close()
