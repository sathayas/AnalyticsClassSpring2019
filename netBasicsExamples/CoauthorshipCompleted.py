import networkx as nx

# reading in the co-authorship information
f = open('Bib-RFTBrainImaging.txt','r', encoding='iso-8859-1')
listListAuthors = []
line = f.readline()
while line:
    listAuthors = line.strip().replace('.','').split(', ')
    listListAuthors.append(listAuthors)
    line = f.readline()
f.close()

# initializing the graph
G = nx.Graph()

# loop over list of list of co-authors
for iList in listListAuthors:
    # loop over list of co-authors
    for iAuthor in iList:
        G.add_node(iAuthor)  # adding an author as a node
        for jAuthor in iList:
            if iAuthor!=jAuthor:
                G.add_edge(iAuthor, jAuthor)  # adding edges


# just for fun, plotting the network
import matplotlib.pyplot as plt

plt.figure(figsize=[8,5])
pos = nx.kamada_kawai_layout(G) # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=50)
nx.draw_networkx_edges(G, pos, edge_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=5, font_color='DarkGreen')
plt.axis('off')
plt.title('Co-authorship network')
plt.show()

    
        
