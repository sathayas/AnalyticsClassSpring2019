import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import girvan_newman, modularity
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


# loading network data
G_football = nx.read_gml('football.gml')  # Football network
y_true = [d['value'] for n, d in G_football.nodes(data=True)]  # true conf labels


# Community detection with the label propagation algorithm
commIndSet_football = label_propagation_communities(G_football)
commInd_football_lp = [list(x) for x in iter(commIndSet_football)]
# generating the detected community labels & adj rand index
y_pred_lp = [n for n in G_football.nodes()]
      # make a node list. Then replace the node name with the predicted
      # communitiy assignment.
for j,jComm in enumerate(commInd_football_lp):
    for k in jComm:
        y_pred_lp[y_pred_lp.index(k)] = j
rand_lp = adjusted_rand_score(y_true,y_pred_lp)


# Community detection with the girvan-newman algorithm
commIndSet_football = girvan_newman_opt(G_football)
commInd_football_gn = [list(x) for x in iter(commIndSet_football)]
# generating the detected community labels & adj rand index
y_pred_gn = [n for n in G_football.nodes()]
      # make a node list. Then replace the node name with the predicted
      # communitiy assignment.
for j,jComm in enumerate(commInd_football_gn):
    for k in jComm:
        y_pred_gn[y_pred_gn.index(k)] = j
rand_gn = adjusted_rand_score(y_true,y_pred_gn)




# drawing the graph (football network)
plt.figure(figsize=[10,4.5])
plt.subplot(131)

# first, graph with true communitities
pos = nx.kamada_kawai_layout(G_football, weight=None) # positions for all nodes
for iComm in range(max(y_true)+1):
    nodeList = [n for n, d in G_football.nodes(data=True) if d['value']==iComm]
    nx.draw_networkx_nodes(G_football, pos, nodelist=nodeList,
                           cmap=plt.cm.rainbow, vmin=0, vmax=max(y_true),
                           node_color = [iComm]*len(nodeList),
                           node_size=100)
nx.draw_networkx_edges(G_football, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_football, pos, font_size=7, font_color='Black')
plt.title('True communities')
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
plt.title('Label propagation\nAdj. Rand index = %6.4f' % rand_lp) 
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
plt.title('Girvan-Newman\nAdj. Rand index = %6.4f' % rand_gn)
plt.axis('off')
plt.xlim([-1.15, 1.15])
plt.ylim([-1.15, 1.15])


plt.subplots_adjust(hspace=0.15, wspace=0.075, bottom=0.025, top=0.875,
                    left=0.05, right=0.95)
plt.show()


