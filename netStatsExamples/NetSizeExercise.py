import networkx as nx
import numpy as np

# loading network data
G_LesMis = nx.read_gml('lesmis.gml')  # Les Miserables
G_football = nx.read_gml('football.gml')  # Football network
G_SP500 = nx.read_gexf('SP500.gexf')  # S&P500
G_facebook = nx.read_edgelist('facebook_combined.edgelist')  # facebook
G_power = nx.read_gml('power.gml', label='id')  # power grid
G_fMRI = nx.read_adjlist('fMRI_HighRes.adjlist')  # fMRI network

# Exercise here!
