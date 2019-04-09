import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import modularity


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
mu = [0.0675, 0.10, 0.20, 0.40]
G_list = []
trueMod_list = []
trueCommInd_list = []
for imu in mu:
    G = LFR_benchmark_graph(n, tau1, tau2, imu, average_degree=5,
                            min_community=25, seed=10)
    G_list.append(G)
    commIndSet = {frozenset(G.nodes[v]['community']) for v in G}
    commInd = [list(x) for x in iter(commIndSet)]
    Q = modularity(G, commInd)
    trueMod_list.append(Q)
    trueCommInd_list.append(commInd)



# communitiy detection by Girvan-Newman method
commInd_list = []
mod_list = []
for i,imu in enumerate(mu):
    commInd = girvan_newman_opt(G_list[i])
    commInd_list.append(commInd)
    Q = modularity(G_list[i], commInd)
    mod_list.append(Q)


# drawing the graph --- Kamada-Kawai layout
plt.figure(figsize=[10,6])
for i,imu in enumerate(mu):
    # true module assignment
    plt.subplot(2,4,i+1)
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
    plt.title('True communities\nprob = %6.4f\nmodularity = %6.4f' % (imu,trueMod_list[i]))
    plt.axis('off')

    # girvan-newman
    plt.subplot(2,4,i+1+4)
    for iComm, iCommInd in enumerate(commInd_list[i]):
        nx.draw_networkx_nodes(G, pos, nodelist=list(iCommInd),
                               cmap=plt.cm.rainbow, vmin=0,
                               vmax=len(commInd_list[i])-1,
                               node_color = [iComm]*len(iCommInd),
                               node_size=50)
    nx.draw_networkx_edges(G, pos, edge_color='lightblue')
    plt.title('Girvan-Newman\nModularity: %6.4f' % mod_list[i])
    plt.axis('off')


plt.subplots_adjust(hspace=0.2, wspace=0.05, bottom=0.025, top=0.875,
                    left=0.05, right=0.95)
plt.show()




# communitiy detection by label propagation method
commInd_list = []
mod_list = []
for i,imu in enumerate(mu):
    commIndSet = label_propagation_communities(G_list[i])
    commInd = [list(x) for x in iter(commIndSet)]
    commInd_list.append(commInd)
    Q = modularity(G_list[i], commInd)
    mod_list.append(Q)


# drawing the graph --- Kamada-Kawai layout
plt.figure(figsize=[10,6])
for i,imu in enumerate(mu):
    # true module assignment
    plt.subplot(2,4,i+1)
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
    plt.title('True communities\nprob = %6.4f\nmodularity = %6.4f' % (imu,trueMod_list[i]))
    plt.axis('off')

    # girvan-newman
    plt.subplot(2,4,i+1+4)
    for iComm, iCommInd in enumerate(commInd_list[i]):
        nx.draw_networkx_nodes(G, pos, nodelist=list(iCommInd),
                               cmap=plt.cm.rainbow, vmin=0,
                               vmax=len(commInd_list[i])-1,
                               node_color = [iComm]*len(iCommInd),
                               node_size=50)
    nx.draw_networkx_edges(G, pos, edge_color='lightblue')
    plt.title('Label propagation\nModularity: %6.4f' % mod_list[i])
    plt.axis('off')


plt.subplots_adjust(hspace=0.2, wspace=0.05, bottom=0.025, top=0.875,
                    left=0.05, right=0.95)
plt.show()

