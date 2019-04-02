import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship



# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[12,6])
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

# drawing the distance from node 1
plt.subplot(132)
D1 = nx.shortest_path_length(G_karate, source=1)
D1_node = list(D1.keys())
D1_d = list(D1.values())
# for loop do draw nodes and labels
nodeC = ['red','magenta','orange','green','blue','purple']
for i,iNode in enumerate(D1_node):
    nodeLabel = {iNode: str(D1_d[i])}
    nx.draw_networkx_nodes(G_karate, pos, nodelist=[iNode], 
                           node_color=nodeC[D1_d[i]])
    nx.draw_networkx_labels(G_karate, pos, labels= nodeLabel,
                            font_size=10, font_color='White')
nx.draw_networkx_edges(G_karate, pos,
                       edge_color='lightblue')
plt.title('Distance from node 1')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])

# drawing the distance from node 17
plt.subplot(133)
D17 = nx.shortest_path_length(G_karate, source=17)
D17_node = list(D17.keys())
D17_d = list(D17.values())
# for loop do draw nodes and labels
nodeC = ['red','magenta','orange','green','blue','purple']
for i,iNode in enumerate(D17_node):
    nodeLabel = {iNode: str(D17_d[i])}
    nx.draw_networkx_nodes(G_karate, pos, nodelist=[iNode], 
                           node_color=nodeC[D17_d[i]])
    nx.draw_networkx_labels(G_karate, pos, labels= nodeLabel,
                            font_size=10, font_color='White')
nx.draw_networkx_edges(G_karate, pos,
                       edge_color='lightblue')
plt.title('Distance from node 17')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])

plt.show()


# closeness centrality
Cclo_karate = nx.closeness_centrality(G_karate)  
Cclo_netsci = nx.closeness_centrality(G_netsci)  



# sorting nodes by closeness centrality
# first, karate network
Cclo_karate_node = Cclo_karate.keys()
Cclo_karate_k = Cclo_karate.values()
sortedNodes_karate = sorted(zip(Cclo_karate_node, Cclo_karate_k), 
                            key=lambda x: x[1], reverse=True)
sCclo_karate_node, sCclo_karate_k = zip(*sortedNodes_karate)

# next, netsci network
Cclo_netsci_node = Cclo_netsci.keys()
Cclo_netsci_k = Cclo_netsci.values()
sortedNodes_netsci = sorted(zip(Cclo_netsci_node, Cclo_netsci_k), 
                            key=lambda x: x[1], reverse=True)
sCclo_netsci_node, sCclo_netsci_k = zip(*sortedNodes_netsci)



# top nodes and their closeness centrality
print('Karate network -- Top degree centrality nodes')
print('Node             \tCloseness centrality')
for iNode in range(5):
    print('%-16s\t' % str(sCclo_karate_node[iNode]), end='')
    print('%6.4f' % sCclo_karate_k[iNode])
print()

print('Network science co-authorship network -- Top degree centrality nodes')
print('Node             \tCloseness centrality')
for iNode in range(15):
    print('%-16s\t' % str(sCclo_netsci_node[iNode]), end='')
    print('%6.4f' % sCclo_netsci_k[iNode])
print()




# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos, 
                       cmap=plt.cm.coolwarm, node_color=list(Cclo_karate_k))
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='White')
plt.axis('off')
plt.title('Karate network\nand closeness centrality')
vmin = sCclo_karate_k[-1]
vmax = sCclo_karate_k[0]
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = plt.colorbar(sm, shrink=0.5)
cbar.ax.set_ylabel('Closeness centrality')
plt.show()

