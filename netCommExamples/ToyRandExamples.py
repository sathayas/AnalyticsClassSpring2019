import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms.community import girvan_newman 
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import modularity
from sklearn.metrics import adjusted_rand_score

# girman-newman method, optimized with modularity
def girvan_newman_opt(G, verbose=False):
    runningMaxMod = 0
    commIndSetFull = girvan_newman(G)
    for iNumComm in range(2,len(G)):
        if verbose:
            print('Commnity detection iteration : %d' % iNumComm)
        iPartition = next(commIndSetFull)  # partition with iNumComm communities
        Q = modularity(G, iPartition)  # modularity
        if Q>runningMaxMod:  # saving the optimum partition and associated info
            runningMaxMod = Q
            OptPartition = iPartition
    return OptPartition



# generating toy networks with different inter-community connection prob
n = 150
tau1 = 3.0
tau2 = 2.0
mu = [0.0675, 0.20]
G_list = []
trueCommInd_list = []
y_true_list = []
for imu in mu:
    G = LFR_benchmark_graph(n, tau1, tau2, imu, average_degree=5,
                            min_community=25, seed=10)
    G_list.append(G)
    commIndSet = {frozenset(G.nodes[v]['community']) for v in G}
    commInd = [list(x) for x in iter(commIndSet)]
    trueCommInd_list.append(commInd)
    # generating the true community labels
    y_true = np.ones(n)
    for i,iComm in enumerate(commInd):
        y_true[iComm] = i
    y_true_list.append(y_true)



# communitiy detection by Girvan-Newman method
commInd_gn_list = []
rand_gn_list = []
for i,imu in enumerate(mu):
    commInd = girvan_newman_opt(G_list[i])
    commInd_gn_list.append(commInd)
    # generating the detected community labels
    y_pred = np.ones(n)
    for j,jComm in enumerate(commInd):
        y_pred[list(jComm)] = j
    ari = adjusted_rand_score(y_true_list[i],y_pred)
    rand_gn_list.append(ari)



# communitiy detection by label propagation
commInd_lp_list = []
rand_lp_list = []
for i,imu in enumerate(mu):
    commIndSet = label_propagation_communities(G_list[i])
    commInd = [list(x) for x in iter(commIndSet)]
    commInd_lp_list.append(commInd)
    # generating the detected community labels
    y_pred = np.ones(n)
    for j,jComm in enumerate(commInd):
        y_pred[list(jComm)] = j
    ari = adjusted_rand_score(y_true_list[i],y_pred)
    rand_lp_list.append(ari)



# drawing the graph --- Kamada-Kawai layout
plt.figure(figsize=[7,9])
for i,imu in enumerate(mu):
    # true module assignment
    plt.subplot(3,2,i+1)
    G = G_list[i]
    commIndSet = {frozenset(G.nodes[v]['community']) for v in G}
    commInd = [list(x) for x in iter(commIndSet)]

    pos = nx.kamada_kawai_layout(G, weight=None) # positions for all nodes
    for iComm in range(len(commInd)):
        nx.draw_networkx_nodes(G, pos, nodelist=commInd[iComm],
                               cmap=plt.cm.rainbow, vmin=0, vmax=len(commInd)-1,
                               node_color = [iComm]*len(commInd[iComm]),
                               node_size=50)
    nx.draw_networkx_edges(G, pos, edge_color='lightblue')
    plt.title('True communities\nprob = %6.4f' % imu)
    plt.axis('off')

    # girvan-newman
    plt.subplot(3,2,i+3)
    for iComm, iCommInd in enumerate(commInd_gn_list[i]):
        nx.draw_networkx_nodes(G, pos, nodelist=list(iCommInd),
                               cmap=plt.cm.rainbow, vmin=0,
                               vmax=len(commInd_gn_list[i])-1,
                               node_color = [iComm]*len(iCommInd),
                               node_size=50)
    nx.draw_networkx_edges(G, pos, edge_color='lightblue')
    plt.title('Girvan-Newman\nAdj. Rand index: %6.4f' % rand_gn_list[i])
    plt.axis('off')

    # label propagation
    plt.subplot(3,2,i+5)
    for iComm, iCommInd in enumerate(commInd_lp_list[i]):
        nx.draw_networkx_nodes(G, pos, nodelist=list(iCommInd),
                               cmap=plt.cm.rainbow, vmin=0,
                               vmax=len(commInd_lp_list[i])-1,
                               node_color = [iComm]*len(iCommInd),
                               node_size=50)
    nx.draw_networkx_edges(G, pos, edge_color='lightblue')
    plt.title('Label propagation\nAdj. Rand index: %6.4f' % rand_lp_list[i])
    plt.axis('off')


plt.subplots_adjust(hspace=0.3, wspace=0.05, bottom=0.025, top=0.925,
                    left=0.05, right=0.95)
plt.show()

