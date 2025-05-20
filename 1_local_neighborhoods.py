import sys
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from itertools import compress
import colorsys

filepath_highd = sys.argv[1] # Filepath to the high dimensional data
outpath = sys.argv[2] # Path to save result to

data_name = input("Please enter your data nickname to save as: ")
n_kmeans = int(input("How many k-means clusters? Must be an integer: "))

n_NN = 5 # for transition points
data_highd = pd.read_csv(filepath_highd, header=None)
num_points = len(data_highd)

# K-means clustering
kmeans = KMeans(n_clusters=n_kmeans, random_state=0).fit(X=data_highd)
nns = NearestNeighbors(n_neighbors=n_NN).fit(data_highd)

# Helper functions 
def get_cluster(point_index, kmeans_data=kmeans):
    # Given the index of a point from the data, returns the index of the cluster it is in.
    return(kmeans_data.labels_[point_index])

def darker_pastel_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(h, 0.5, 0.75) for h in hues]

# For plotting
if n_kmeans == 5:
    colors = ['blueviolet', 'deepskyblue', 'aquamarine', 'orange', 'red']
elif n_kmeans == 10:
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
elif n_kmeans == 20:
    colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]
else:
    colors = darker_pastel_colors(n_kmeans)

# Get neighbor matrices
neighbor_indices_matrix = nns.kneighbors(return_distance=False)
neighbor_clusters_matrix = np.array([get_cluster(x) for x in neighbor_indices_matrix])

# Get all transition points
transition_bool_vector = np.zeros(num_points)
for row_index in range(num_points):
    if len(np.unique(neighbor_clusters_matrix[row_index])) > 1:
        transition_bool_vector[row_index] = 1

set_transition_dict = {cluster_index: set() for cluster_index in range(n_kmeans)}
for row_index in range(num_points):
    if len(np.unique(neighbor_clusters_matrix[row_index])) > 1:
        curr_cluster = get_cluster(row_index)
        transition_cluster_indices = np.setdiff1d(neighbor_clusters_matrix[row_index], np.array(curr_cluster))
        for n in transition_cluster_indices:
            set_transition_dict[n].add(row_index)

transition_dict = {key: list(value) for key, value in set_transition_dict.items()} # maps cluster to list of point indices of transition points
transition_network_dict = {key: list(map(get_cluster, value)) for key, value in transition_dict.items()} # maps cluster to list of which cluster the transition point belongs to

# Draw network graph
atlas_network = nx.Graph()
for v in range(n_kmeans):
    atlas_network.add_node(v)
max_weight = 0
for v in range(n_kmeans):
    for neighbor in np.unique(transition_network_dict[v]):
        edge_weight = transition_network_dict[v].count(neighbor)
        if atlas_network.has_edge(v, neighbor):
            new_weight = edge_weight + atlas_network.edges[v,neighbor]['weight']
            atlas_network.edges[v,neighbor]['weight'] = new_weight
            if new_weight > max_weight:
                max_weight = new_weight
        else:
            atlas_network.add_edge(v, neighbor, weight=edge_weight)
            if edge_weight > max_weight:
                max_weight = edge_weight

pos = nx.spring_layout(atlas_network, seed=7, k=1.5) # adjust k to change node spacing. default is 1/sqrt(n)
nx.draw_networkx_nodes(atlas_network, pos, node_size = 350, node_color=colors)
for edge in atlas_network.edges(data='weight'):
    nx.draw_networkx_edges(atlas_network, pos, edgelist=[edge], width=edge[2]*20/max_weight)
# nx.draw_networkx_labels(atlas_network, pos, font_size=11, font_family="sans-serif")
plt.savefig(outpath+data_name+'_transition_network.png', dpi=300)

try:
    os.mkdir(outpath+'clusters')
    cluster_out = os.path.join(outpath, 'clusters')
except FileExistsError:
    cluster_out = os.path.join(outpath, 'clusters')
except Exception as e:
    print(f"An error occurred: {e}")

# Add transition points to clusters they are neighbors of
border_points = dict()
clusters = []
for cluster_index in range(n_kmeans):
    cluster_data = data_highd.iloc[kmeans.labels_ == cluster_index,:]
    cluster_data = pd.concat([cluster_data, data_highd.iloc[transition_dict[cluster_index],:]])
    clusters.append(cluster_data)
    cluster_data.to_csv(os.path.join(cluster_out, data_name + "_highd_deepatlas_cluster_"+str(cluster_index)+".csv"), header=False,index=False)
    all_transitions = np.concatenate((transition_bool_vector[kmeans.labels_ == cluster_index], np.array([1 for i in range(len(transition_dict[cluster_index]))])))
    border_points[cluster_index] = list(compress(range(len(all_transitions)), all_transitions)) #get indices for which all_transitions is 1 (black)


with open(outpath + data_name + '_transition_dict.json','w') as fp:
    fp.write(json.dumps(border_points))