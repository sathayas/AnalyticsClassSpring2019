import networkx as nx
import numpy as np

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship


# Network sizes
print('Network sizes')
print("Zachary's karate network, n:", len(G_karate.nodes()), sep='')
print("Zachary's karate network, m:", len(G_karate.edges()), sep='')

print("Network science co-authorship network, n:",
      len(G_netsci.nodes()), sep='')
print("Network science co-authorship network, m:",
      len(G_netsci.edges()), sep='')
print()

# Giant component sizes
print('Giant component sizes')
GC_karate = len(sorted(nx.connected_components(G_karate), key = len, reverse=True)[0])
GC_netsci = len(sorted(nx.connected_components(G_netsci), key = len, reverse=True)[0])
print("Zachary's karate network, GC:", GC_karate, sep='')
print("Network science co-authorship network, GC:",
      GC_netsci, sep='')
print()

# Relative giant component sizes
rGC_karate = GC_karate/len(G_karate.nodes())
rGC_netsci = GC_netsci/len(G_netsci.nodes())
print('Relative giant component sizes')
print('Zachary\'s karate network, GC: %4.2f' % rGC_karate)
print("Network science co-authorship network, GC: %4.2f" % rGC_netsci)


