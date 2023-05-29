# CONFIRMATORY TETRAD TEST
import random
import numpy as np
from kl_div_elim import S_theta
import matplotlib.pyplot as plt
from scipy.stats import chi2


# tetrad test
def statistic(obs_matrix):
    # -Step 1: Define tau estimations
    sigma = np.cov(obs_matrix, rowvar=False)
    tau = np.zeros(3)
    tau[0] = sigma[0][1] * sigma[2][3] - sigma[0][2] * sigma[1][3]  # τ_1234
    tau[1] = sigma[0][2] * sigma[3][1] - sigma[0][3] * sigma[2][1]  # τ_1342
    tau[2] = sigma[0][3] * sigma[1][2] - sigma[0][1] * sigma[3][2]  # τ_1423

    # -Step 2: Compute estimate cov matrix of tau: D(σ)'Cov(σ)D(σ)

    # mapping for τ: (0, 1, 2) --> (0, 1, 2, 3)
    mapping_tau = {}
    mapping_tau[0] = (0, 1, 2, 3)
    mapping_tau[1] = (0, 2, 3, 1)
    mapping_tau[2] = (0, 3, 1, 2)

    # mapping for σ
    n = 0
    mapping_sigma = {}  # index to pair
    mapping_b = {}  # pair to index
    for i in range(4):
        for j in range(i + 1, 4):
            mapping_sigma[n] = (i, j)
            mapping_b[i, j] = n
            n += 1

    def partial_der(m_tau, m_sigma):
        if m_sigma[0] == m_tau[0] and m_sigma[1] == m_tau[1] or m_sigma[0] == m_tau[1] and m_sigma[1] == m_tau[0]:
            der = sigma[m_tau[2], m_tau[3]]
        elif m_sigma[0] == m_tau[2] and m_sigma[1] == m_tau[3] or m_sigma[0] == m_tau[3] and m_sigma[1] == m_tau[2]:
            der = sigma[m_tau[0], m_tau[1]]
        elif m_sigma[0] == m_tau[0] and m_sigma[1] == m_tau[2] or m_sigma[0] == m_tau[2] and m_sigma[1] == m_tau[0]:
            der = -sigma[m_tau[1], m_tau[3]]
        elif m_sigma[0] == m_tau[1] and m_sigma[1] == m_tau[3] or m_sigma[0] == m_tau[3] and m_sigma[1] == m_tau[1]:
            der = -sigma[m_tau[0], m_tau[2]]
        else:
            der = 0
        return der

    # gradient matrix D(σ)
    D_s = np.zeros((3, 6))
    for i in range(3):
        for j in range(6):
            D_s[i, j] = partial_der(mapping_tau[i], mapping_sigma[j])

    # Cov(σ): cov(σ_ij, σ_kl) = σ_ik * σ_jl + σ_il * σ_jk
    def cov_s(matrix, ind1, ind2):
        cov = matrix[ind1[0], ind2[0]] * matrix[ind1[1], ind2[1]] - matrix[ind1[0], ind2[1]] * matrix[ind1[1], ind2[0]]
        return cov

    cov_sigma = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            cov_sigma[i, j] = cov_s(sigma, mapping_sigma[i], mapping_sigma[j])

    cov_tau = D_s @ cov_sigma @ D_s.T  # cov matrix of tau
    return sigma, tau, cov_tau, D_s


# -Step 3,4: Calculate test statistic (original + alternatives)
def T_stat_0(X):
    sigma, tau, cov_tau, D_s = statistic(X)
    choices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    A = np.random.choice(range(len(choices)), size=2, replace=False)  # selecting non-redundant vanishing tetrads
    cov_tau_1 = np.linalg.inv(A @ cov_tau @ A.T)
    T_0 = len(X) * tau.T @ A.T @ cov_tau_1 @ A @ tau
    return T_0


def T_stat_1(X):
    sigma, tau, cov_tau, D_s = statistic(X)
    diag_cov_tet = np.diag(np.diag(cov_tau))
    T_1 = len(X) * tau.T @ diag_cov_tet.T @ tau
    return T_1


def T_stat_2(X):
    sigma, tau, cov_tau, D_s = statistic(X)
    T_2 = len(X) * tau.T @ tau
    return T_2


def T_stat_3(X):
    sigma, tau, cov_tau, D_s = statistic(X)
    diag_cov_tet = np.diag(np.diag(cov_tau))
    choices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    A = np.random.choice(range(len(choices)), size=2, replace=False)  # selecting non-redundant vanishing tetrads
    cov_tau_1 = np.linalg.inv(A @ diag_cov_tet @ A.T)
    T_3 = len(X) * tau.T @ A.T @ cov_tau_1 @ A @ tau
    return T_3


# Bootstrapping
def bootstrap(X, T_stat, B, N):
    T = T_stat(X)  # tetrad test for observed matrix
    S_theta_array = S_theta.detach().numpy()  # tensor with gradient --> numpy array
    R = np.linalg.cholesky(S_theta_array)  # S_theta=RR'
    Q = np.linalg.cholesky(statistic(X)[0])  # X=QQ'
    Q_1 = np.linalg.inv(Q)
    Z = X @ Q_1.T @ R.T  # Z=X(Q^-1)'R'

    T_b = np.zeros(B)
    for b in range(B):
        idx_bs = np.random.choice(N, N)
        Z_b = Z[idx_bs]
        T_b[b] = T_stat(Z_b)  # tetrad test for B simulation realizations
    p_val = (T_b >= T).mean()
    return p_val, T, T_b


if __name__ == "__main__":  # ----------------------------------------------------------------------------- #

    N = 1000  # observations
    D = 4  # variables
    B = 300  # simulation realizations of bootstrap
    niter = 2000

    # ------------------------ GENERATE DATA / CALCULATE TEST STATISTICS & P-VALUES ------------------------ #

    T_0 = np.zeros(niter)
    p_value_0 = np.zeros(niter)
    p_value_1 = np.zeros(niter)
    p_value_2 = np.zeros(niter)
    p_value_3 = np.zeros(niter)

    # MATRIX SAMPLED FROM MODEL
    # L_rand = np.tanh(np.random.randn(D))
    L = [-0.8707149, 0.78428483, 0.49303985, -0.27977438]  # fixed L sampled like L_rand
    Z_0 = np.random.randn(N)
    X = np.zeros((N, D))

    for s in range(niter):  # niter different X (coming from same seed)
        for d in range(D):
            epsilon = 1 - L[d] ** 2
            noise = np.random.randn(N) * epsilon ** .5
            X[:, d] = L[d] * Z_0 + noise  # observation matrix
        T_0[s] = T_stat_0(X)
        p_value_0[s] = 1 - chi2.cdf(T_0[s], 2)  # p-value for T_0
        p_value_1[s], T_1, T_b1 = bootstrap(X, T_stat_1, B, N)  # p-value for T_1
        p_value_2[s], T_2, T_b2 = bootstrap(X, T_stat_2, B, N)  # p-value for T_2
        p_value_3[s], T_3, T_b3 = bootstrap(X, T_stat_3, B, N)  # p-value for T_3

    # MATRIX SAMPLED from a MVN distribution with given covariance/correlation matrix C (generated at random)
    for s in range(niter):  # niter different X (coming from same seed)
        A = np.random.randn(D, D)
        A_mul = np.matmul(A, A.T)
        Cov_data = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                Cov_data[i][j] = A_mul[i][j] / (np.sqrt(A_mul[i][i]) * np.sqrt(A_mul[j][j]))
        X_random = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=Cov_data, size=N)  # random obs matrix
        T_0[s] = T_stat_0(X)
        p_value_0[s] = 1 - chi2.cdf(T_0[s], 2)  # p-value for T_0
        p_value_1[s], T_1, T_b1 = bootstrap(X, T_stat_1, B, N)  # p-value for T_1
        p_value_2[s], T_2, T_b2 = bootstrap(X, T_stat_2, B, N)  # p-value for T_2
        p_value_3[s], T_3, T_b3 = bootstrap(X, T_stat_3, B, N)  # p-value for T_3


    # -------------------------------------------- PLOT FIGURES -------------------------------------------- #
    # PLOTTING T_0 & PDF OF chi^2 for multiple obs matrices X_i
    for s in range(niter):
        x = np.arange(0, 20, 0.001)
        plt.figure(1)
        plt.clf()
        plt.hist(T_0[s], density=True, label='T_0', bins=20, histtype='step')
        plt.plot(x, chi2.pdf(x, df=2), color='red', linewidth=2, label='χ^2')
        plt.title(f'T_stat_0 for {niter} observation matrices X_i (i=1,...,{niter})')
        plt.xlabel(f"tetrad test statistic (N={N})")
        leg = plt.legend(loc='upper right')
    plt.show()
    plt.close()



    # PLOTTING p-values for multiple obs matrices X_i
    for s in range(niter):
        plt.figure(1)
        plt.clf()
        plt.hist(p_value_0[s], bins=30, label='p-value($T_0$)')  # histtype='step', alpha=1)
        # plt.hist(p_value_1, bins=30, label='p-value($T_1$)', histtype='step', alpha=1)
        # plt.hist(p_value_2, bins=30, label='p-value($T_2$)', histtype='step', alpha=1)
        # plt.hist(p_value_3, bins=30, label='p-value($T_3$)', histtype='step', alpha=1)
        leg = plt.legend(loc='upper right')
        plt.title(f'Tetrad test statistic for observation matrices X_i', fontsize=16)
        # plt.xlabel(f"p-value (M={niter}, N={N}, B={B})", fontsize=14)
        plt.xlabel(f"p-value (M={niter}, N={N})", fontsize=14)
        plt.xlim([0, 1])
        # plt.ylim([0, 200])
    plt.show()
    plt.close()

    # PLOTTING T_1 & T_b1 values
    plt.figure(2)
    plt.clf()
    plt.axvline(x=T_1, color='r', linestyle='-', label='T')
    plt.hist(T_b1, label='T_b', bins=50)
    leg = plt.legend(loc='upper right')
    plt.title(f'T_stat_1 (p_value = {round(p_value_1[s], 2)})', fontsize=16)
    plt.ylabel("simulation realizations", fontsize=14)
    plt.xlabel("tetrad test statistic", fontsize=14)
    plt.show()
    plt.close()

    # COMPARE p-values of alternative test statistics for multiple obs matrices X_i
    for s in range(niter):
        plt.figure(3)
        plt.clf()
        plt.hist(p_value_1[s], bins=30, label='p-value($T_1$)', histtype='stepfilled', alpha=0.6)
        plt.hist(p_value_2[s], bins=30, label='p-value($T_2$)', histtype='stepfilled', alpha=0.6)
        plt.hist(p_value_3[s], bins=30, label='p-value($T_3$)', histtype='stepfilled', alpha=0.6)
        leg = plt.legend(loc='upper right')
        plt.title(f'Tetrad test statistic for observation matrices X_i', fontsize=16)
        plt.xlabel(f"p-value (M={niter}, N={N}, B={B})", fontsize=14)
        plt.xlim([0, 1])
    plt.show()
    plt.close()

    # COMPARE p-values of original test T_0 & bootstrapped T_1 for multiple obs matrices X_i
    for s in range(niter):
        plt.figure(4)
        plt.clf()
        plt.hist(p_value_0[s], bins=50, label='p-value($T_0$)', histtype='stepfilled', alpha=0.6, ec='black')
        plt.hist(p_value_1[s], bins=50, label='p-value(bootstrapped_$T_1$)', histtype='stepfilled', alpha=0.6, ec='black')
        leg = plt.legend(loc='upper right')
        plt.title(f'Tetrad test statistic for observation matrices X_i', fontsize=16)
        plt.xlabel(f"p-value (M={niter}, N={N}, B={B})", fontsize=14)
        plt.xlim([0, 1])
    plt.show()
    plt.close()

    # SCATTER of p-values of all test statistics (multiple X)
    for s in range(niter):
        plt.figure(5)  # scatter for T_1 & T_2
        plt.clf()
        plt.title(f'Correlation of p-values for test statistics $T_1$ and $T_2$', fontsize=16)
        plt.xlabel(f"p-value ($T_1$)", fontsize=14)
        plt.ylabel(f"p-value ($T_2$)", fontsize=14)
        # add x=y line
        plt.scatter(p_value_1[s], p_value_2[s], alpha=0.6)
        plt.plot([0, 1], [0, 1], color='red')

        plt.figure(6)  # scatter for T_2 & T_3
        plt.clf()
        plt.title(f'Correlation of p-values for test statistics $T_2$ and $T_3$', fontsize=16)
        plt.xlabel(f"p-value ($T_2$)", fontsize=14)
        plt.ylabel(f"p-value ($T_3$)", fontsize=14)
        # add x=y line
        plt.scatter(p_value_2[s], p_value_3[s], alpha=0.6)
        plt.plot([0, 1], [0, 1], color='red')

        plt.figure(7)  # scatter for T_1 & T_3
        plt.clf()
        plt.title(f'Correlation of p-values for test statistics $T_1$ and $T_3$', fontsize=16)
        plt.xlabel(f"p-value ($T_1$)", fontsize=14)
        plt.ylabel(f"p-value ($T_3$)", fontsize=14)
        # add x=y line
        plt.scatter(p_value_1[s], p_value_3[s], alpha=0.6)
        plt.plot([0, 1], [0, 1], color='red')

    plt.show()
    plt.close()
