import networkx as nx
import matplotlib.pyplot as plt

# first, creating a graph
G=nx.Graph()

# adding nodes
G.add_node(1)  # single node
G.add_nodes_from([2,3])  # multiple nodes
G.add_nodes_from(list(range(3,8)))
G.add_node('Boss')

# a list of nodes
print(G.nodes())


# adding edges
G.add_edge(1,2)  # single edge
G.add_edges_from([(1,2),(1,3),(5,7)])  # multiple edges
for iEdge in range(1,7):
    G.add_edge(iEdge,iEdge+1)
for i in G.nodes():
    if i!='Boss':
        G.add_edge('Boss',i)

# a list of edges
print(G.edges())

# which nodes is Boss connected to
print(G['Boss'])

# how about node 3
print(G[3])



# drawing the graph
nx.draw(G, with_labels=True)
plt.show()



# edge weights
# first, all edge weights are set to 1
for iEdge in G.edges():
    G.edges[iEdge]['weight']=1.0
# giving some edges more weight
G['Boss'][3]['weight'] = 3
G[2][3]['weight'] = 4
G['Boss'][7]['weight'] = 4

# checking edge weights
print(G.edges(data=True))



# drawing the graph  --- circular layout
pos = nx.circular_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos)

# edges
edgeweight = [ d['weight'] for (u,v,d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=edgeweight)

# labels
nx.draw_networkx_labels(G, pos)

plt.axis('off')
plt.show()



# drawing the graph  --- Kamada-Kawai layout
pos = nx.kamada_kawai_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos)

# edges
nx.draw_networkx_edges(G, pos, width=edgeweight)

# labels
nx.draw_networkx_labels(G, pos)

plt.axis('off')
plt.show()

