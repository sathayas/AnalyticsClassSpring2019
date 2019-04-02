import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship

# degree centrality
Cdeg_karate = nx.degree_centrality(G_karate)  
Cdeg_netsci = nx.degree_centrality(G_netsci)  



# sorting nodes by degree centrality
# first, karate network
Cdeg_karate_node = Cdeg_karate.keys()
Cdeg_karate_k = Cdeg_karate.values()
sortedNodes_karate = sorted(zip(Cdeg_karate_node, Cdeg_karate_k), 
                            key=lambda x: x[1], reverse=True)
sCdeg_karate_node, sCdeg_karate_k = zip(*sortedNodes_karate)

# next, netsci network
Cdeg_netsci_node = Cdeg_netsci.keys()
Cdeg_netsci_k = Cdeg_netsci.values()
sortedNodes_netsci = sorted(zip(Cdeg_netsci_node, Cdeg_netsci_k), 
                            key=lambda x: x[1], reverse=True)
sCdeg_netsci_node, sCdeg_netsci_k = zip(*sortedNodes_netsci)



# top nodes and their degree centrality
print('Karate network -- Top degree centrality nodes')
print('Node           \tDegree centrality')
for iNode in range(5):
    print('%-14s\t' % str(sCdeg_karate_node[iNode]), end='')
    print('%6.4f' % sCdeg_karate_k[iNode])
print()

print('Network science co-authorship network -- Top degree centrality nodes')
print('%-16s' % 'Node' + '\tDegree centrality')
for iNode in range(15):
    print('%-16s\t' % str(sCdeg_netsci_node[iNode]), end='')
    print('%6.4f' % sCdeg_netsci_k[iNode])
print()



# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos, 
                       cmap=plt.cm.coolwarm, node_color=list(Cdeg_karate_k))
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='White')
plt.axis('off')
plt.title('Karate network\nand degree centrality')
vmin = sCdeg_karate_k[-1]
vmax = sCdeg_karate_k[0]
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = plt.colorbar(sm, shrink=0.5)
cbar.ax.set_ylabel('Degree centrality')
plt.show()

