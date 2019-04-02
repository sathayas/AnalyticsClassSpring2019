import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship

# eigenvector centrality
Ceig_karate = nx.eigenvector_centrality(G_karate)  
Ceig_netsci = nx.eigenvector_centrality(G_netsci)  



# sorting nodes by eigenvector centrality
# first, karate network
Ceig_karate_node = Ceig_karate.keys()
Ceig_karate_k = Ceig_karate.values()
sortedNodes_karate = sorted(zip(Ceig_karate_node, Ceig_karate_k), 
                            key=lambda x: x[1], reverse=True)
sCeig_karate_node, sCeig_karate_k = zip(*sortedNodes_karate)

# next, netsci network
Ceig_netsci_node = Ceig_netsci.keys()
Ceig_netsci_k = Ceig_netsci.values()
sortedNodes_netsci = sorted(zip(Ceig_netsci_node, Ceig_netsci_k), 
                            key=lambda x: x[1], reverse=True)
sCeig_netsci_node, sCeig_netsci_k = zip(*sortedNodes_netsci)



# top nodes and their eigenvector centrality
print('Karate network -- Top degree centrality nodes')
print('Node           \tEigenvector centrality')
for iNode in range(5):
    print('%-14s\t' % str(sCeig_karate_node[iNode]), end='')
    print('%6.4f' % sCeig_karate_k[iNode])
print()

print('Network science co-authorship network -- Top degree centrality nodes')
print('%-16s' % 'Node' + '\tEigenvector centrality')
for iNode in range(15):
    print('%-16s\t' % str(sCeig_netsci_node[iNode]), end='')
    print('%6.4f' % sCeig_netsci_k[iNode])
print()




# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos, 
                       cmap=plt.cm.coolwarm, node_color=list(Ceig_karate_k))
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='White')
plt.axis('off')
plt.title('Karate network\nand eigenvector centrality')
vmin = sCeig_karate_k[-1]
vmax = sCeig_karate_k[0]
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = plt.colorbar(sm, shrink=0.5)
cbar.ax.set_ylabel('Eigenvector centrality')
plt.show()

