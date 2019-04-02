import networkx as nx
import numpy as np

# loading network data
G_LesMis = nx.read_gml('lesmis.gml')  # Les Miserables
G_facebook = nx.read_edgelist('facebook_combined.edgelist')  # facebook


# calculating and printing clustering coefficient
print('Les Mis network: %6.4f' % nx.average_clustering(G_LesMis))
print('Facebook network: %6.4f' % nx.average_clustering(G_facebook))
