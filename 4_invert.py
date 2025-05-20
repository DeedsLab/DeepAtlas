import sys
import os
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import seaborn as sns
import tensorflow as tf

from numpy import arctanh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import keras
from keras.models import load_model
from pynverse import inversefunc

# Note that current code assumes use of adj_tanh activation function and PCA into 2 dimensions

directory = sys.argv[1] # Where kmeans data is located (clusters directory)
json_path = sys.argv[2] # Where the transition json is located
models_path = sys.argv[3] # Where neural network models are located (models directory)

n_NN_radius = int(input("Points will be sampled from a radius of the nth nearest neighbor. What value of n to use? Must be an integer: ")) 

def adj_tanh(x):
    return tf.math.tanh(x) + 0.1*x

def inverse_adj_tanh(x):
    return inversefunc(adj_tanh)(x)

act_func = inverse_adj_tanh

def get_jds(original_data, lower_data, k=20):
    def neighbors_helper(data, k):
        # for a given dataset, finds the k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        return nbrs.kneighbors(return_distance=False)

    def jaccard_helper(A,B):
        # for two sets A and B, finds the Jaccard distance J between A and B
        A = set(A)
        B = set(B)
        union = list(A|B)
        intersection = list(A & B)
        J = (len(union) - len(intersection))/len(union)
        return J

    high_D_neighborhood = neighbors_helper(original_data, k)
    low_D_neighborhood = neighbors_helper(lower_data, k)

    jaccard_distances=[]
    for i in range(len(original_data.index)):
        jaccard_distances.append(jaccard_helper(low_D_neighborhood[i,:],high_D_neighborhood[i,:]))
    
    return jaccard_distances

# Create a dictionary of the models, where the key is the string of the cluster number and the value is the model path
model_dict = dict()
for model_file in os.listdir(models_path):
    f = os.path.join(models_path, model_file)
    if os.path.isfile(f):
        cluster = f.split('_')[-2]
        model_dict[cluster] = f

num_clusts = len(model_dict)

# List of the filenames of the neural network outputs
file_list = list()
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        file_list.append(f)

with open(json_path) as f:
    realpts_transition_dict = json.load(f)

sampled_transition_dict = {f.split('_')[-1].split('.')[0] : list() for f in file_list}

# Colors for plotting purposes
def darker_pastel_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(h, 0.5, 0.75) for h in hues]

if num_clusts == 5:
    colors = ['blueviolet', 'deepskyblue', 'aquamarine', 'orange', 'red']
elif num_clusts == 10:
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
elif num_clusts == 20:
    colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]
else:
    colors = darker_pastel_colors(num_clusts)

def point_on_sphere(p, k):
    # Samples a point from the sphere around p of radius k
    d = len(p)
    v = [random.uniform(-1,1) for _ in range(d)]
    v = v / np.linalg.norm(v)
    p_prime = p + k * v
    return p_prime

# Set up structures for tracking
nn_AJD_per_cluster = list()
pc_AJD_per_cluster = list()
btwn_AJD_per_cluster = list()
cluster_data = dict()
out_model_dict = dict()

# Note: Default PCA to 2D
pca = PCA(n_components=2, svd_solver='full')

# Get lower embedding and sample new data from that lower dimensional space
for file in file_list:
    data = pd.read_csv(file, header=None)
    pca_data = pd.DataFrame(pca.fit_transform(data))
    num_points = data.shape[0] 
    sampled_data = pd.DataFrame(index=range(num_points), columns=range(data.shape[1]))
    cluster = file.split('_')[-1].split('.')[0]
    model = load_model(model_dict[str(cluster)], custom_objects={"adj_tanh": adj_tanh}) 
    out_data = pd.DataFrame(index=range(num_points), columns=range(data.shape[1]))

    # If user chose a radius from which to sample. Otherwise, use direct inverse.
    if n_NN_radius > 0:
        for point_index in range(num_points):
            pt = np.array(data.iloc[point_index,]).reshape((1,3))
            out_data.iloc[point_index,] = model.predict(pt)
        out_model_dict[cluster] = out_data

        nns = NearestNeighbors(n_neighbors=n_NN_radius).fit(out_data)
        neighbor_distances, neighbor_indices = nns.kneighbors()
        for point_index in range(num_points):
            point = point_index
            r = neighbor_distances[point][n_NN_radius-1] # distance of random real point to its n_NNth neighbor
            X = out_data.iloc[point,]
            z = point_on_sphere(X, r)
            if point in realpts_transition_dict[cluster]: # if sampled point that this point is based on is a transition point, mark it as such
                sampled_transition_dict[cluster].append(point_index)
            sampled_data.iloc[point_index] = z
        cluster_data[cluster] = sampled_data
    else:
        cluster_data[cluster] = out_data

    nn_AJD_per_cluster.append(np.mean(get_jds(data, out_data)))
    pc_AJD_per_cluster.append(np.mean(get_jds(data, pca_data)))
    btwn_AJD_per_cluster.append(np.mean(get_jds(out_data, pca_data)))

# Plot sampled data to sanity check
for cluster in range(num_clusts):
    color = colors[cluster]
    to_plot = cluster_data[str(cluster)]
    cmap = cm.LinearSegmentedColormap.from_list("", list((color, 'black')))
    border_bool = [int(x in sampled_transition_dict[str(cluster)]) for x in range(to_plot.shape[0])]
    fig, ax = plt.subplots()
    plt.scatter(out_model_dict[str(cluster)].iloc[:,0], out_model_dict[str(cluster)].iloc[:,1], color = color)
    plt.show()

# Load models to use for inverting
model_dict = dict()
for model_file in os.listdir(models_path):
    f = os.path.join(models_path, model_file)
    if os.path.isfile(f):
        cluster = f.split('_')[-2]
        model_dict[cluster] = f

def invert(x, w, b, activation = arctanh):
    # Mathematically invert a standard Dense layer
    return np.dot(activation(x) - b,np.linalg.inv(w))

all_inverted_sampled = dict()
all_inverted_modeled = dict()

# Get inverse function from model structure and apply to each cluster
for cluster in range(len(file_list)):
    # modeled will be the direct inverse and sampled will be the new data sampled based on the lower dimensional embedding
    model = load_model(model_dict[str(cluster)], custom_objects={"adj_tanh": adj_tanh})
    modeled = out_model_dict[str(cluster)]
    sampled = cluster_data[str(cluster)]
    inv_data_sampled = pd.DataFrame(index=range(sampled.shape[0]), columns=range(sampled.shape[1]))
    inv_data_modeled = pd.DataFrame(index=range(modeled.shape[0]), columns=range(modeled.shape[1]))
    for point in range(len(sampled)):
        inv_sampled = np.dot((sampled.iloc[point,] - model.layers[-1].bias), np.linalg.inv(model.layers[-1].weights[0])) # last layer doesn't use an activation function
        inv_modeled = np.dot((modeled.iloc[point,] - model.layers[-1].bias), np.linalg.inv(model.layers[-1].weights[0])) # last layer doesn't use an activation function

        for l in range(len(model.layers)-1, 0, -1): # loop backwards
            layer = l-1 # zero indexed
            if 'dropout' in model.layers[layer].name:
                pass
            else:
                weight = model.layers[layer].weights[0]
                bias = model.layers[layer].bias
                inv_sampled = invert(inv_sampled, weight, bias, act_func)
                inv_modeled = invert(inv_modeled, weight, bias, act_func)
        inv_data_sampled.iloc[point,] = inv_sampled
        inv_data_modeled.iloc[point,] = inv_modeled

    all_inverted_sampled[cluster] = inv_data_sampled
    all_inverted_modeled[cluster] = inv_data_modeled

    ajd = np.mean(get_jds(inv_data_sampled, sampled))

# Plot the result of passing the sampled data through the inverted model
fig, ax = plt.subplots()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.grid(False)
ax3D.set_xticks([])
ax3D.set_yticks([])
ax3D.set_zticks([])

for clust in range(len(file_list)):
    df = all_inverted_sampled[clust]
    color = colors[clust]
    ax3D.scatter(df[0], df[1], df[2], label=clust, color=color)
plt.title('Inverse of Data Sampled from the '+str(n_NN_radius)+ 'Nearest Neighbor')
plt.axis('off')
plt.show()

# Plot the direct inverse
fig, ax = plt.subplots()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.grid(False)
ax3D.set_xticks([])
ax3D.set_yticks([])
ax3D.set_zticks([])

for clust in range(len(file_list)):
    df = all_inverted_modeled[clust]
    color = colors[clust]
    ax3D.scatter(df[0], df[1], df[2], label=clust, color=color)
plt.title('Direct Inverse')
plt.axis('off')
plt.show()