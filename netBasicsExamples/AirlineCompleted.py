import networkx as nx
import pandas as pd

# loading the data
routeData = pd.read_csv('Southwest_Mar2013.csv')


# first, creating a graph
G=nx.Graph()

# loop over routes
for iRoute in range(len(routeData)):
    # adding nodes
    origNode = routeData.loc[iRoute,'ORIGIN']
    destNode = routeData.loc[iRoute,'DEST']
    G.add_node(origNode)
    G.add_node(destNode)
    # adding an edge 
    G.add_edge(origNode,destNode)


# just for fun, plotting the network
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=[8,5])
pos = nx.kamada_kawai_layout(G) # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=50)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=5, font_color='DarkGreen')
plt.axis('off')
plt.title('Southwest Airlines network')
plt.show()

    
