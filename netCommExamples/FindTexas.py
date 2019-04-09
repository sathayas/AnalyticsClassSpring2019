import networkx as nx
import numpy as np
from networkx.algorithms.community import label_propagation_communities
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


# loading network data
G_football = nx.read_gml('football.gml')  # Football network


# Community detection with the label propagation algorithm
commIndSet_football = label_propagation_communities(G_football)
commInd_football_lp = [list(x) for x in iter(commIndSet_football)]


# Community detection with the girvan-newman algorithm
commInd_football_gn = girvan_newman_opt(G_football)



# Finding the module for Texas (label propagation)
print('Label propagation:')
for iComm in commInd_football_lp:
    if 'Texas' in iComm:
        for iSchool in iComm:
            print(iSchool)
print()


# Finding the module for Texas (Girvan-Newman)
print('Girvan-Newman:')
for iComm in commInd_football_gn:
    if 'Texas' in iComm:
        for iSchool in iComm:
            print(iSchool)
print()
