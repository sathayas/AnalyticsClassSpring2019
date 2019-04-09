import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import girvan_newman, modularity


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



# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_football = nx.read_gml('football.gml')  # Football network


# Community detection with the label propagation algorithm
commIndSet_karate = label_propagation_communities(G_karate)
commInd_karate_lp = [list(x) for x in iter(commIndSet_karate)]

commIndSet_football = label_propagation_communities(G_football)
commInd_football_lp = [list(x) for x in iter(commIndSet_football)]


# Community detection with the girvan-newman algorithm
commInd_karate_gn = girvan_newman_opt(G_karate)
commInd_football_gn = girvan_newman_opt(G_football)



# drawing the graph (karate network)
plt.figure(figsize=[9,4])

# first, graph without community assignments
plt.subplot(131)
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_edges(G_karate, pos,
                       edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.title('Original karate network')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])

# next, graph with communities in different colors (label propagation)
plt.subplot(132)
for iComm in range(len(commInd_karate_lp)):
    nx.draw_networkx_nodes(G_karate, pos, nodelist=commInd_karate_lp[iComm],
                           cmap=plt.cm.rainbow,
                           vmin=0, vmax=len(commInd_karate_lp)-1,
                           node_color = [iComm]*len(commInd_karate_lp[iComm]),
                           node_size=300)
nx.draw_networkx_edges(G_karate, pos,
                       edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='White')
plt.title('Label propagation')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])

# (girvan-newman)
plt.subplot(133)
for iComm in range(len(commInd_karate_gn)):
    nx.draw_networkx_nodes(G_karate, pos, nodelist=commInd_karate_gn[iComm],
                           cmap=plt.cm.rainbow,
                           vmin=0, vmax=len(commInd_karate_gn)-1,
                           node_color = [iComm]*len(commInd_karate_gn[iComm]),
                           node_size=300)
nx.draw_networkx_edges(G_karate, pos,
                       edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='White')
plt.title('Girvan-Newman')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])

plt.subplots_adjust(hspace=0.15, wspace=0.075, bottom=0.025, top=0.875,
                    left=0.05, right=0.95)
plt.show()



# drawing the graph (football network)
plt.figure(figsize=[10,4.5])
plt.subplot(131)

# first, graph without community assignments
pos = nx.kamada_kawai_layout(G_football, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_football, pos, node_size=100)
nx.draw_networkx_edges(G_football, pos,
                       edge_color='lightblue')
nx.draw_networkx_labels(G_football, pos, font_size=7, font_color='Black')
plt.title('Original football network')
plt.axis('off')
plt.xlim([-1.15, 1.15])
plt.ylim([-1.15, 1.15])

# next, graph with communities in different colors (label propagation)
plt.subplot(132)
for iComm in range(len(commInd_football_lp)):
    nx.draw_networkx_nodes(G_football, pos, nodelist=commInd_football_lp[iComm],
                           cmap=plt.cm.rainbow,
                           vmin=0, vmax=len(commInd_football_lp)-1,
                           node_color = [iComm]*len(commInd_football_lp[iComm]),
                           node_size=100)
nx.draw_networkx_edges(G_football, pos,
                       edge_color='lightblue')
nx.draw_networkx_labels(G_football, pos, font_size=7, font_color='Black')
plt.title('Label propagation')
plt.axis('off')
plt.xlim([-1.15, 1.15])
plt.ylim([-1.15, 1.15])

# (girvan-newman)
plt.subplot(133)
for iComm in range(len(commInd_football_gn)):
    nx.draw_networkx_nodes(G_football, pos, nodelist=commInd_football_gn[iComm],
                           cmap=plt.cm.rainbow,
                           vmin=0, vmax=len(commInd_football_gn)-1,
                           node_color = [iComm]*len(commInd_football_gn[iComm]),
                           node_size=100)
nx.draw_networkx_edges(G_football, pos,
                       edge_color='lightblue')
nx.draw_networkx_labels(G_football, pos, font_size=7, font_color='Black')
plt.title('Girvan-Newman')
plt.axis('off')
plt.xlim([-1.15, 1.15])
plt.ylim([-1.15, 1.15])


plt.subplots_adjust(hspace=0.15, wspace=0.075, bottom=0.025, top=0.875,
                    left=0.05, right=0.95)
plt.show()


