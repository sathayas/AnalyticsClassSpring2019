import networkx as nx
import matplotlib.pyplot as plt

# loading the Les Miserables network
G = nx.read_gml('lesmis.gml')

# drawing the graph  --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='DarkGreen')
plt.axis('off')
plt.title('Les Miserables interaction network')
plt.show()


# loading the college football network
G = nx.read_gml('football.gml')

# drawing the graph  --- Kamada-Kawai layout
plt.figure(figsize=[12,12])
pos = nx.kamada_kawai_layout(G, weight=None) # positions for all nodes
# extracting conference information
conf = []
for i,d in G.nodes(data=True):
    conf.append(d['value'])
# drawing nodes, different conferences in different colors
for iConf in range(12):
    nx.draw_networkx_nodes(G, pos,
                           cmap=plt.cm.tab20, node_color=conf)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='DarkGreen')
plt.axis('off')
plt.title('College football network')
plt.show()


# loading the Florentine family network
G = nx.read_pajek('Padgett.paj')

# drawing the graph  --- Kamada-Kawai layout
plt.figure(figsize=[8,8])
pos = nx.kamada_kawai_layout(G, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='DarkGreen')
plt.axis('off')
plt.title('Florentine family network')
plt.show()


# loading the Chesapeake Bay food web
G = nx.read_pajek('Chesapeake.paj')

# drawing the graph  --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_color='Salmon', node_size=100)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='DarkGreen')
plt.axis('off')
plt.title('Chesapeake Bay food web')
plt.show()


# loading the C Elegans neural network
G = nx.read_gml('celegansneural.gml')

# drawing the graph  --- random
plt.figure(figsize=[10,10])
pos = nx.random_layout(G) # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=30, node_color='salmon')
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
plt.axis('off')
plt.title('C Elegans neural network')
plt.show()

