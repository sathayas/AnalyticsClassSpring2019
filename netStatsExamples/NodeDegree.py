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


# degree sequence
G_karate.degree()
G_netsci.degree()


# average node degree
k_karate = [d for n, d in G_karate.degree()]
k_netsci = [d for n, d in G_netsci.degree()]
np.mean(k_karate)
np.mean(k_netsci)


# degree distribution
plt.hist(k_karate,bins=20)
plt.title('Degree distribution, karate')
plt.show()

plt.hist(k_netsci,bins=30)
plt.title('Degree distribution, netsci')
plt.show()


# degree distribution, against ranks, log-log
sk_netsci = sorted(k_netsci, reverse=True)
rank_netsci = np.arange(len(sk_netsci)) + 1
plt.plot(sk_netsci,rank_netsci,'b-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Rank')
plt.title('Degree distribution, netsci')
plt.show()


# Connection probability matrix
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
Rjk_karate = nx.degree_mixing_matrix(G_karate)
plt.imshow(Rjk_karate)
plt.xlabel('Node 1 degree')
plt.ylabel('Node 2 degree')
plt.colorbar()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
Rjk_netsci = nx.degree_mixing_matrix(G_netsci)
plt.imshow(Rjk_netsci)
plt.xlabel('Node 1 degree')
plt.ylabel('Node 2 degree')
plt.colorbar()
plt.show()



# assortativity coefficient
nx.degree_assortativity_coefficient(G_karate)
nx.degree_assortativity_coefficient(G_netsci)
