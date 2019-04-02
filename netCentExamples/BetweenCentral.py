import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship

# betweenness centrality
Cbet_karate = nx.betweenness_centrality(G_karate)  
Cbet_netsci = nx.betweenness_centrality(G_netsci)  



# sorting nodes by betweenness centrality
# first, karate network
Cbet_karate_node = Cbet_karate.keys()
Cbet_karate_k = Cbet_karate.values()
sortedNodes_karate = sorted(zip(Cbet_karate_node, Cbet_karate_k), 
                            key=lambda x: x[1], reverse=True)
sCbet_karate_node, sCbet_karate_k = zip(*sortedNodes_karate)

# next, netsci network
Cbet_netsci_node = Cbet_netsci.keys()
Cbet_netsci_k = Cbet_netsci.values()
sortedNodes_netsci = sorted(zip(Cbet_netsci_node, Cbet_netsci_k), 
                            key=lambda x: x[1], reverse=True)
sCbet_netsci_node, sCbet_netsci_k = zip(*sortedNodes_netsci)



# top nodes and their betweenness centrality
print('Karate network -- Top degree centrality nodes')
print('Node             \tBetweenness centrality')
for iNode in range(5):
    print('%-16s\t' % str(sCbet_karate_node[iNode]), end='')
    print('%6.4f' % sCbet_karate_k[iNode])
print()

print('Network science co-authorship network -- Top degree centrality nodes')
print('Node             \tBetweenness centrality')
for iNode in range(15):
    print('%-16s\t' % str(sCbet_netsci_node[iNode]), end='')
    print('%6.4f' % sCbet_netsci_k[iNode])
print()




# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos, 
                       cmap=plt.cm.coolwarm, node_color=list(Cbet_karate_k))
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='White')
plt.axis('off')
plt.title('Karate network\nand betweenness centrality')
vmin = sCbet_karate_k[-1]
vmax = sCbet_karate_k[0]
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = plt.colorbar(sm, shrink=0.5)
cbar.ax.set_ylabel('Betweenness centrality')
plt.show()

