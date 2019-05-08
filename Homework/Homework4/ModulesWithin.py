import networkx as nx
import numpy as np
from networkx.algorithms.community import girvan_newman, modularity


# girman-newman method, optimized with modularity
def girvan_newman_opt(G, verbose=False):
    runningMaxMod = 0
    commIndSetFull = girvan_newman(G)
    for iNumComm in range(2,len(G)):
        if verbose:
            print('Commnity detection iteration : %d' % iNumComm)
        iPartition = next(commIndSetFull)  # partition with iNumComm communities
        Q = modularity(G, iPartition)  # modularity
        if Q>runningMaxMod:  # saving the optimum partition and associated info
            runningMaxMod = Q
            OptPartition = iPartition
    return OptPartition


##### Loading the network data, extracting the giant component
G = nx.read_gml('netscience.gml')
GC = max(nx.connected_component_subgraphs(G), key=len)

##### modular partition via girvan-newman
commInd = girvan_newman_opt(GC)

##### Calculating modularity and number of modules
print('Number of modules: %d' % len(commInd))
print('Modularity: %6.4f' % modularity(GC, commInd))


##### extracting the largest module as a network
maxCommNodes = max(commInd, key=lambda coll: len(coll))
print('Largest module size: %d' % len(maxCommNodes))
G_maxComm = G.subgraph(maxCommNodes)


##### sub-modular partition of the largest module
commInd_maxComm = girvan_newman_opt(G_maxComm)
mod_maxComm = modularity(G_maxComm, commInd_maxComm)
nComm_maxComm = len(commInd_maxComm)

print('Number of sub-modules: %d' % nComm_maxComm)
print('Modularity of sub-modular partition: %6.4f' % mod_maxComm)
