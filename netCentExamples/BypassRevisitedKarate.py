import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network

# getting the position of nodes in this network, to be used later
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes

# adding edges
G_karate.add_edges_from([(2,24), (6,9), (14,29)])

# closeness centrality
Cclo_karate = nx.closeness_centrality(G_karate)  

# sorting nodes by closeness centrality
Cclo_karate_node = Cclo_karate.keys()
Cclo_karate_k = Cclo_karate.values()
sortedNodes_karate = sorted(zip(Cclo_karate_node, Cclo_karate_k), 
                            key=lambda x: x[1], reverse=True)
sCclo_karate_node, sCclo_karate_k = zip(*sortedNodes_karate)


# top nodes and their closeness centrality
print('Karate network -- Top degree centrality nodes')
print('Node             \tCloseness centrality')
for iNode in range(5):
    print('%-16s\t' % str(sCclo_karate_node[iNode]), end='')
    print('%6.4f' % sCclo_karate_k[iNode])
print()



# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[6,6])

nx.draw_networkx_nodes(G_karate, pos, 
                       cmap=plt.cm.coolwarm, node_color=list(Cclo_karate_k))
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='White')
plt.axis('off')
plt.title('Karate network\nand closeness centrality\n(with bypasses)')
vmin = sCclo_karate_k[-1]
vmax = sCclo_karate_k[0]
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = plt.colorbar(sm, shrink=0.5)
cbar.ax.set_ylabel('Closeness centrality')
plt.show()
