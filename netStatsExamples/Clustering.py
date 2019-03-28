import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship


# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[12,6])
eListB = [(7,17), (17,6),(6,7)]
eListG = [(7,1), (1,5), (5,7)]
eListP = [(7,1), (1,6), (6,7)]
plt.subplot(131)
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_nodes(G_karate, pos, nodelist=[7], node_color='DarkOrange')
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_edges(G_karate, pos, edgelist=eListB,
                       edge_color='blue', width=3.0)
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.title('Triangle 1')
plt.axis('off')

plt.subplot(132)
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_nodes(G_karate, pos, nodelist=[7], node_color='DarkOrange')
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_edges(G_karate, pos, edgelist=eListG,
                       edge_color='green', width=3.0)
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.title('Triangle 2')
plt.axis('off')

plt.subplot(133)
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_nodes(G_karate, pos, nodelist=[7], node_color='DarkOrange')
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_edges(G_karate, pos, edgelist=eListP,
                       edge_color='purple', width=3.0)
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.title('Triangle 3')
plt.axis('off')

plt.subplots_adjust(hspace=0.00, wspace=0.00, left=0.05, right=0.95)
plt.show()



# Clustering coefficient at each node
nx.clustering(G_karate)
nx.clustering(G_netsci)



# Average clustering coefficient for the network
nx.average_clustering(G_karate)
nx.average_clustering(G_netsci)

