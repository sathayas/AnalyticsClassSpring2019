import networkx as nx
import numpy as np
from random import seed, randint
import matplotlib.pyplot as plt

# loading network data
G_karate = nx.read_gml('karate.gml', label='id')  # Karate network
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship


def print_netStats(G):
    '''
    priting out network statistics

    Input parameter:
        G:    A graph object
    
    Returns:
        None

    The following stats are printed for G: 
        GC:    Relative giant component size
        L:     Average shortest path lengths (for giant component only)
        D:     Diamter (for giant component only)
        C:     Average clustering coefficient
    '''
    GConly = max(nx.connected_component_subgraphs(G), key=len)
    GC = len(GConly)/len(G)
    L = nx.average_shortest_path_length(GConly)
    D = nx.diameter(GConly)
    C = nx.average_clustering(G)
    print('GC size: %5.3f' % GC, end='')
    print('\t\tL : %5.2f' % L, end='')
    print('\t\tD : %2d' % D, end='')
    print('\t\tC : %4.2f' % C)


# initializing the random number generator
seed(0)

#### removing nodes randomly (karate)
G_karate_randdel = G_karate.copy()
print('Before node removal')
print_netStats(G_karate_randdel)
# for loop for node removal
for iRemove in range(5):
    nodeOut = randint(1,len(G_karate))
    while nodeOut not in G_karate_randdel:
        nodeOut = randint(1,len(G_karate))
    print('Removing node ', nodeOut)
    G_karate_randdel.remove_node(nodeOut)
    print_netStats(G_karate_randdel)


#### removing nodes randomly (netsci)
G_netsci_randdel = G_netsci.copy()
nodeList = list(G_netsci.nodes())
print('Before node removal')
print_netStats(G_netsci_randdel)
# for loop for node removal
for iRemove in range(15):
    nodeOut = nodeList[randint(1,len(G_netsci))]
    while nodeOut not in G_netsci_randdel:
        nodeOut = nodeList[randint(1,len(G_netsci))]
    print('Removing node ', nodeOut)
    G_netsci_randdel.remove_node(nodeOut)
    print_netStats(G_netsci_randdel)




# finding the node with high degree (karate)
k_karate = [d for n, d in G_karate.degree()]
node_karate = [n for n, d in G_karate.degree()]
sortedNodes = sorted(zip(node_karate, k_karate), key=lambda x: x[1],
                     reverse=True)
snode_karate, sk_karate = zip(*sortedNodes)



# finding the node with high degree (netsci)
k_netsci = [d for n, d in G_netsci.degree()]
node_netsci = [n for n, d in G_netsci.degree()]
sortedNodes = sorted(zip(node_netsci, k_netsci), key=lambda x: x[1],
                     reverse=True)
snode_netsci, sk_netsci = zip(*sortedNodes)




#### removing highly connected nodes first (karate)
G_karate_attack = G_karate.copy()
print('Before node removal')
print_netStats(G_karate_attack)
# for loop for node removal
for iRemove in range(5):
    nodeOut = snode_karate[iRemove]
    print('Removing node ', nodeOut)
    G_karate_attack.remove_node(nodeOut)
    print_netStats(G_karate_attack)



#### removing highly connected nodes first (netsci)
G_netsci_attack = G_netsci.copy()
print('Before node removal')
print_netStats(G_netsci_attack)
# for loop for node removal
for iRemove in range(15):
    nodeOut = snode_netsci[iRemove]
    print('Removing node ', nodeOut)
    G_netsci_attack.remove_node(nodeOut)
    print_netStats(G_netsci_attack)



# drawing the graph (karate network only) --- Kamada-Kawai layout
plt.figure(figsize=[12,6])
plt.subplot(131)
pos = nx.kamada_kawai_layout(G_karate, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_karate, pos)
nx.draw_networkx_edges(G_karate, pos,
                       edge_color='lightblue')
nx.draw_networkx_labels(G_karate, pos, font_size=10, font_color='DarkGreen')
plt.title('Original karate network')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])


plt.subplot(132)
nodeList = G_karate_randdel.nodes()
edgeList = G_karate_randdel.edges()
nx.draw_networkx_nodes(G_karate, pos, nodelist=nodeList)
nx.draw_networkx_edges(G_karate, pos, edgelist=edgeList,
                       edge_color='lightblue')
plt.title('Random deletion, karate network')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])


plt.subplot(133)
nodeList = G_karate_attack.nodes()
edgeList = G_karate_attack.edges()
nx.draw_networkx_nodes(G_karate, pos, nodelist=nodeList)
nx.draw_networkx_edges(G_karate, pos, edgelist=edgeList,
                       edge_color='lightblue')
plt.title('Targeted attack, karate network')
plt.axis('off')
plt.xlim([-0.6, 0.65])
plt.ylim([-0.85, 1.2])

plt.subplots_adjust(hspace=0.00, wspace=0.00, left=0.05, right=0.95)
plt.show()




