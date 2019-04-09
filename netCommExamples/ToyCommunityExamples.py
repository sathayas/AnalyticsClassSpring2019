import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import LFR_benchmark_graph


# first, a simple toy example of modular network
n = 150
tau1 = 3.0
tau2 = 2.0
mu = 0.0675
G_list = []
G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5,
                        min_community=25, seed=10)



# drawing the graph --- Kamada-Kawai layout
# without community assignment
plt.figure(figsize=[8,6])

plt.subplot(121)
pos = nx.kamada_kawai_layout(G, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=50)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
plt.title('Toy network with communities')
plt.axis('off')

# extracting community assignment indices
commIndSet = {frozenset(G.nodes[v]['community']) for v in G}
commInd = [list(x) for x in iter(commIndSet)]

# drawing with community assignment
plt.subplot(122)
for iComm in range(len(commInd)):
    nx.draw_networkx_nodes(G, pos, nodelist=commInd[iComm],
                           cmap=plt.cm.rainbow, vmin=0, vmax=len(commInd)-1,
                           node_color = [iComm]*len(commInd[iComm]),
                           node_size=50)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
plt.title('Toy network with commnitites\nin different colors')
plt.axis('off')

plt.subplots_adjust(hspace=0.15, wspace=0.075, bottom=0.025, top=0.875,
                    left=0.05, right=0.95)
plt.show()



# generating toy networks with different inter-community connection prob
n = 150
tau1 = 3.0
tau2 = 2.0
mu = [0.0675, 0.10, 0.20, 0.40]
G_list = []
for imu in mu:
    G = LFR_benchmark_graph(n, tau1, tau2, imu, average_degree=5,
                            min_community=25, seed=10)
    G_list.append(G)

    
    

# drawing the graph --- Kamada-Kawai layout
plt.figure(figsize=[8,8])
for i,imu in enumerate(mu):
    plt.subplot(2,2,i+1)
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
    plt.title('Inter-community connection probability\n%6.4f' % imu)
    plt.axis('off')

plt.subplots_adjust(hspace=0.15, wspace=0.075, bottom=0.025, top=0.9,
                    left=0.05, right=0.95)
plt.show()
