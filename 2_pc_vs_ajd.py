import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from csv import writer
import matplotlib.pyplot as plt
import matplotlib.colors as cm

directory = sys.argv[1] # Where kmeans data is located (clusters directory)
outpath = sys.argv[2] # Path to save result to

data_name = input("Please enter your data nickname to save as: ")

# Get cluster filenames
file_list = list()
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f): 
        file_list.append(f)

# Function to calculate Jaccard distances between a high dimensional dataset and its lower dimensional embedding
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

# Calculate the AJD of clusters at each PC dimension up to 2,000 or the max of the dataset
pcvsajd = []
for file in file_list:
    data = pd.read_csv(file, header=None)
    cluster = file.split('_')[-1].split('.')[0]
    pc_max = 2000 if min(data.shape) > 2000 else min(data.shape) 
    k = 20
    if pc_max < 20:
        k = pc_max - 1

    ajds = []
    ajds.append(cluster)
    for pca_components in range(1, pc_max+1): 
        pca = PCA(n_components=pca_components)
        pca_representation = pca.fit_transform(data)
        ajd = np.mean(get_jds(data, pca_representation, k))
        ajds.append(ajd)
    pcvsajd.append(ajds)

    with open(outpath + data_name + 'pcvsajd.csv','a') as fd:
        writer_obj = writer(fd)    
        writer_obj.writerow(ajds)
        fd.close()

# Plot the results
ajd_data = pd.DataFrame(pcvsajd)
ajd_data = ajd_data.set_index(0).rename_axis(None)

if len(file_list) == 5:
    colors = ['blueviolet', 'deepskyblue', 'aquamarine', 'orange', 'red']
elif len(file_list) == 10:
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
elif len(file_list) == 20:
    colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]
else:
    colors = "grey"

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots()
ajd_data.T.plot(ax=ax, color=colors, alpha=0.7, legend=False)
average_line = ajd_data.mean(axis=0)
average_line.plot(ax=ax, color='black', linewidth=2, label='Average')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
ax.set_xlabel('PC')
ax.set_ylabel('AJD')
ax.set_title(data_name)
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
plt.show()