import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from networkx.algorithms.community import modularity
import json


##### Loading the data into a graph
G = nx.Graph()
f = open('email-Eu-core.txt','r')
while True:
    line = f.readline().strip()
    if not line:
        break
    else:
        nodeList = line.split(' ')
        nodes = [int(x) for x in nodeList]
        G.add_nodes_from(nodes)
        G.add_edge(nodes[0],nodes[-1])

f.close()

###### Number of nodes and edges
print('Number of nodes: %d' % len(G.nodes()))
print('Number of edges: %d' % len(G.edges()))

###### Plotting the degree distribution
k = [d for n, d in G.degree()]
sk = sorted(k, reverse=True)
rank_sk = np.arange(len(sk)) + 1
plt.plot(sk,rank_sk,'b-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Rank')
plt.title('Degree distribution, e-mail network')
plt.show()


###### Closeness centrality, sorted
# closeness centrality
Cclo = nx.closeness_centrality(G)
# sorting nodes by closeness centrality
Cclo_node = Cclo.keys()
Cclo_k = Cclo.values()
sortedNodes = sorted(zip(Cclo_node, Cclo_k),
                        key=lambda x: x[1], reverse=True)
sCclo_node, sCclo_k = zip(*sortedNodes)
# printing out top 10 highest closeness centrality nodes
for i in range(10):
    print('Node: %5d' % sCclo_node[i] + '\t closeness: %6.4f' % sCclo_k[i])



###### reading in true and estimated partitions
# first, true partition
fTrue = open('email-Eu-core-department-labels.txt', 'r')
line = fTrue.readline()
partition_True = []
while line:
    lineData = [int(x) for x in line.strip().split(' ')]
    partition_True.append(lineData)
    line = fTrue.readline()
fTrue.close()
partition_True = sorted(partition_True)
y_True = np.array(partition_True)[:,1]
# then Louvain partiton
fLouvain = open('EmailPartitionLouvain.json','r')
jsonData = json.load(fLouvain)
fLouvain.close()
partition_Louvain = [(int(key), val) for key,val in jsonData.items()]
partition_Louvain = sorted(partition_Louvain)
y_Louvain = np.array(partition_Louvain)[:,1]



####### Modularity and ARI
# converting to list of lists
listList_True = []
for iMod in np.unique(y_True):
    listNodes = [node for node,module in partition_True if module==iMod]
    listList_True.append(listNodes)
listList_Louvain = []
for iMod in np.unique(y_Louvain):
    listNodes = [node for node,module in partition_Louvain if module==iMod]
    listList_Louvain.append(listNodes)

print('Modularity, true modular partition: %6.4f' % modularity(G, listList_True))
print('Modularity, Louvain modular partition: %6.4f' % modularity(G, listList_Louvain))

# Adjusted rand score
print('Adjusted Rand Score: %6.4f' % adjusted_rand_score(y_True,y_Louvain))
