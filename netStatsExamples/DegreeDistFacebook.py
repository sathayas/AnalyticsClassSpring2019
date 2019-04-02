import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading network data
G_facebook = nx.read_edgelist('facebook_combined.edgelist')  # facebook


# degree distribution, against ranks, log-log
k_facebook = [d for n, d in G_facebook.degree()]
sk_facebook = sorted(k_facebook, reverse=True)
rank_facebook = np.arange(len(sk_facebook)) + 1
plt.plot(sk_facebook,rank_facebook,'b-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Rank')
plt.title('Degree distribution, facebook')
plt.show()
