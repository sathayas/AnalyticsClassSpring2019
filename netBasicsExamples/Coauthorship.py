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

# Exercise code here!
