import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship

# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.axis('off')
plt.title('Karate network')
plt.show()



# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
eList = [(31,33), (33,32)]
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_edges(G_karate, pos, edgelist=eList,
                       edge_color='blue', width=3.0)
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.axis('off')
plt.title('Karate network\nShortest path between 31 and 32\n(Path length 2)')
plt.show()



# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
eList = [(17,7), (7,1), (1,9), (9,33), (33,21)]
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_edges(G_karate, pos, edge_color='lightblue')
nx.draw_networkx_edges(G_karate, pos, edgelist=eList,
                       edge_color='blue', width=3.0)
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.axis('off')
plt.title('Karate network\nShortest path between 17 and 21\n(Path length 5)')
plt.show()



# average shortest path lengths
nx.average_shortest_path_length(G_karate)
# the giant component only for the netsci network
G_netsci_GConly = max(nx.connected_component_subgraphs(G_netsci), key=len)
nx.average_shortest_path_length(G_netsci_GConly)


# diameter
nx.diameter(G_karate)
nx.diameter(G_netsci_GConly)  # giant component only for netsci network

