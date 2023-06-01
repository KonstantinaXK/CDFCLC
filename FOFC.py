# FIND ONE FACTOR CLUSTERS
import random
import numpy as np
from scipy.stats import chi2
from itertools import combinations
from testing_algorithm import statistic, T_stat_0, T_stat_1, T_stat_2, T_stat_3, bootstrap
from functools import reduce
import networkx as nx
import matplotlib.pyplot as plt


# ------------------------------------------- STEP 1 OF FOFC ------------------------------------------- #
def FindPureClusters(vrbl, X, alpha=0.05):
    triple_list = list(combinations(vrbl, 3))
    PureList = []
    for triple in triple_list:
        pure = True
        for v in [item for item in vrbl if item not in triple]:
            quartet = triple + (v,)
            p_value_1 = PassesTest(X, quartet)
            if p_value_1 < alpha/(len(vrbl)-3):  # if triple forms vanishing quartet then break
                pure = False
                break
        if pure:
            if triple not in PureList:
                PureList.append(triple)
    print('PureList:', PureList)
    return PureList


def PassesTest(X, quartet):  # test of vanishing tetrad constraints
    subX = X[:, quartet] # sub-matrix of X for current 4 variables
    T_0 = T_stat_0(subX)
    p_value_1 = 1 - chi2.cdf(T_0, 2)  # p-value for T_0
    return p_value_1


# ------------------------------------------- STEP 2 OF FOFC ------------------------------------------- #
def GrowClusters(PureList, par):
    union = reduce(np.union1d, PureList)  # union of triples in PureList
    CList = PureList.copy()  # initialize CList to PureList

    for cluster in CList:
        sub = list(combinations(cluster, 2))
        for u in [item for item in union if item not in cluster]:
            acc = 0
            rej = 0
            for s in sub:
                test_cluster = np.union1d(s, u)
                if tuple(test_cluster) in PureList:
                    acc += 1
                else:
                    rej += 1
            if acc/(rej + acc) >= par:
                CList.append(tuple(np.union1d(cluster, u)))
                sub_list = sub_lists(np.union1d(cluster, u))
                for c in sub_list:
                    if tuple(c) in CList:
                        CList.remove(tuple(c))
    print('CList:', CList)
    return CList


def sub_lists(l):  # function to generate all the sub lists of a list
    comb = []
    for i in range(len(l)):
        comb += [list(j) for j in combinations(l, i)]
    return comb


# ------------------------------------------- STEP 3 OF FOFC ------------------------------------------- #
def SelectedClusters(cluster_list):
    cluster_list.sort(key=len, reverse=True)  # sorting list from longest tuple to shortest
    cluster_list_new = cluster_list.copy()
    for ic, c in enumerate(cluster_list[:-1]):
        if c in cluster_list_new:
            for l in cluster_list[ic+1:]:
                if tuple(set(c) & set(l)):  # delete clusters intersecting with c
                    if l in cluster_list_new:
                        cluster_list_new.remove(l)
    return cluster_list_new  # results in list of non-intersecting clusters


if __name__ == "__main__":  # ----------------------------------------------------------------------------- #
    N = 10000  # observations
    B = 100  # simulation realizations of bootstrap
    vrbl = [i for i in range(10)]  # number of all variables


# ----------------------- GENERATE DATA ----------------------- #
    inaccurate_clusters = 0  # number fo inaccurate clusters
    ind = []  # indicator of inaccurate cluster
    for i in range(500):  # testing for 500 obs matrices
        Xdata = np.zeros((N, 10))  # obs matrix for all variables
        Xdata1 = np.zeros((N, 5))
        Xdata2 = np.zeros((N, 5))

        # Z1 causes Z2
        Z_1 = np.random.randn(N)
        L0 = np.random.rand(1) * 2 - 1
        L0 = L0 * .5 + .5 * np.sign(L0)
        epsilon = 1 - L0 ** 2
        noise = np.random.randn(N) * epsilon ** .5
        Z_2 = L0 * Z_1 + noise

        L1 = np.random.rand(5) * 2 - 1
        L1 = L1*.5 + .5 * np.sign(L1)
        for d in range(5):
            epsilon = 1 - L1[d] ** 2
            noise = np.random.randn(N) * epsilon ** .5
            Xdata1[:, d] = L1[d] * Z_1 + noise

        L2 = np.random.rand(5) * 2 - 1
        L2 = L2 * .5 + .5 * np.sign(L2)
        for d in range(5):
            epsilon = 1 - L2[d] ** 2
            noise = np.random.randn(N) * epsilon ** .5
            Xdata2[:, d] = L2[d] * Z_2 + noise

        Xdata = np.hstack((Xdata1, Xdata2))  # merged obs matrix
        Xdata -= Xdata.mean(axis=0, keepdims=True)
        Xdata /= Xdata.std(axis=0, keepdims=True)

# --------------------------------------------------------------------- #
        PureList = FindPureClusters(vrbl, Xdata, alpha=0.05)
        CList = GrowClusters(PureList, par=0.7)
        Sel_Clusters = SelectedClusters(CList)
        print('Non-intersecting Clusters:', Sel_Clusters)
        if Sel_Clusters != [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)]:
            inaccurate_clusters += 1
            ind.append(i)

# ------------------------ PLOT GRAPH OF CLUSTERS ------------------------ #
        plt.figure()
        edges = []
        for cluster in Sel_Clusters:
            for j in cluster:
                edges.append((f'$L_{Sel_Clusters.index(cluster)}$', f'$X_{j}$'))
        G = nx.DiGraph(directed=True)
        G.add_edges_from(edges)
        options = {
            'node_color': 'orange',
            'node_size': 700,
            'width': 1,
        }
        graph = nx.draw_networkx(G, **options, pos=nx.kamada_kawai_layout(G))
    plt.show()
    plt.close()
    print('Inaccurate clusters=', inaccurate_clusters)
    print(ind)

