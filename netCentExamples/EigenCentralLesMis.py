import networkx as nx
import numpy as np

# loading network data
G_LesMis = nx.read_gml('lesmis.gml')  # Les Miserables

# eigenvector centraltiy
Ceig_LesMis = nx.eigenvector_centrality(G_LesMis)


# sorting nodes by eigenvector centrality
Ceig_LesMis_node = Ceig_LesMis.keys()
Ceig_LesMis_k = Ceig_LesMis.values()
sortedNodes_LesMis = sorted(zip(Ceig_LesMis_node, Ceig_LesMis_k), 
                            key=lambda x: x[1], reverse=True)
sCeig_LesMis_node, sCeig_LesMis_k = zip(*sortedNodes_LesMis)


# top nodes and their eigenvector centrality
print('Les Miserables network -- Top eigenvector centrality nodes')
print('%-16s' % 'Node' + '\tEigenvector centrality')
for iNode in range(5):
    print('%-16s\t' % str(sCeig_LesMis_node[iNode]), end='')
    print('%6.4f' % sCeig_LesMis_k[iNode])
print()

