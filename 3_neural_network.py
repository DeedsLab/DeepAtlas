import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger


directory = sys.argv[1] # Where kmeans data is located (clusters directory)
outpath = sys.argv[2] # Path to save result to


def adj_tanh(x):
    # Adjusted tanh function such that the inverse will always be defined
    return tf.math.tanh(x) + 0.1*x

epochs = int(input("How many epochs? Must be an integer: ")) 
num_layers = int(input("How many layers? Must be an integer: "))
pc_d = int(input("What PC dimension to embed into? Must be an integer: "))
input_activation = input("What activation function to use? We recommend tanh or adj_tanh: ")
k_splits = int(input("What value of K for Kfold cross validation? Must be an integer >= 2: "))

try:
    activ_func = tf.keras.activations.get(input_activation)
except ValueError:
    activ_func = locals()[input_activation]
else:
    print("Activation function not found.")

def get_jds(original_data, lower_data, k=20):
    # Function to get the jaccard distances of all points in a dataset.
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

# Get cluster files
file_list = list()
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f): 
        file_list.append(f)
num_clusts = len(file_list)

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

# Create folder to save models to
try:
    os.mkdir(outpath+'models')
    models_out = os.path.join(outpath, 'models')
except FileExistsError:
    models_out = os.path.join(outpath, 'models')
except Exception as e:
    print(f"An error occurred: {e}")

pca = PCA(n_components=pc_d, svd_solver='full')

nn_AJD_per_cluster = list()
pc_AJD_per_cluster = list()
btwn_AJD_per_cluster = list()
cross_vals_per_cluster = list()

for cluster in range(num_clusts):
    data_highd = pd.read_csv(file_list[cluster], header=None)
    data_pca = pd.DataFrame(pca.fit_transform(data_highd))
    high_d = data_highd.shape[1]

    # Pad with zeros
    for diff in range(high_d - pc_d):
        data_pca[pc_d+diff] = np.zeros(data_pca.shape[0])    
    color = colors[cluster]

    btwn = 1
    cross_vals = list()
    kFold = KFold(n_splits=k_splits)
    attempt_count = 0

    # Split data
    for train, test in kFold.split(data_highd):
        X_train = data_highd.iloc[train]
        X_test = data_highd.iloc[test]
        y_train = data_pca.iloc[train]
        y_test = data_pca.iloc[test]

        # btwn is the AJD between the current prediction and the PCA output. If training is taking an extremely long time, increase this value closer to 1 to relax the requirement.
        while btwn > .5:
            attempt_count += 1
            if attempt_count > 50:
                print("Warning: Neural network training has been attempted " + str(attempt_count) + " times. We recommend re-running with more epochs or an increased btwn value.")
            
            # Train neural network model
            model=tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(X_train.shape[1],)))
            for i in range(num_layers):
                model.add(layers.Dense(high_d,activation=activ_func))
            model.add(layers.Dense(high_d))
            model.compile(optimizer='adam', loss=losses.MeanSquaredError())
            checkpoint_path = outpath+"model_results_"+str(cluster) + "_"+color+ "/cp-{epoch:01d}.keras"
            csv_logger= CSVLogger(outpath+"trial_log_"+str(cluster) + "_"+color+".log")

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=False, 
                save_freq=100)
            history = model.fit(X_train, y_train, epochs=epochs, shuffle=True, validation_data=(X_test,y_test), batch_size=100, callbacks=[csv_logger, model_checkpoint_callback])
            model.save(os.path.join(models_out, "best_model_"+str(cluster) + "_"+color+".keras"))

            final_prediction = pd.DataFrame(model.predict(data_highd))
            k = 20 if data_highd.shape[0] > 20 else data_highd.shape[0] - 1

            nn_jds = get_jds(data_highd, final_prediction, k)
            pca_jds = get_jds(data_highd, data_pca, k)
            nn_ajd = np.mean(nn_jds)
            pca_ajd = np.mean(pca_jds)
            btwn_ajd = np.mean(get_jds(final_prediction, data_pca, k))

            btwn = btwn_ajd

        cross_vals.append(mean_squared_error(y_test, model.predict(X_test)))
        nn_AJD_per_cluster.append(nn_ajd)
        pc_AJD_per_cluster.append(pca_ajd)
        btwn_AJD_per_cluster.append(btwn_ajd)
    cross_vals_per_cluster.append(cross_vals)

    # Plot training
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training of cluster: ' + str(cluster))
    plt.legend()
    plt.savefig(outpath+"cluster_" + str(cluster) + "_training.png")

    # Plot the PCA embedding and an overlay of both the PCA embedding and the final neural network prediction
    if pc_d == 2:
        fig, ax = plt.subplots()
        ax.scatter(data_pca[0], data_pca[1], c='black', alpha=.5, label='PCA')
        ax.scatter(final_prediction[0], final_prediction[1], c=color, alpha=.5, label='NN')
        plt.title('Overlayed\n AJD Between: ' + str(btwn_ajd))
        plt.show()

# Plot violin plot of AJDs to evaluate performance
fig, ax = plt.subplots()
ajd_df = pd.DataFrame(
{'nn_ajds': nn_AJD_per_cluster,
'pc_ajds': pc_AJD_per_cluster,
'btwn_ajds': btwn_AJD_per_cluster})
sns.violinplot(ajd_df, cut=0)
plt.ylabel('AJD')
plt.savefig(outpath+"ajd_violinplt.png")

print("avg_nn_AJD_per_layer:" + str(np.mean(nn_AJD_per_cluster)))
print("avg_pc_AJD_per_layer:" + str(np.mean(pc_AJD_per_cluster)))

# Plot MSE over cross validation runs
fig, ax = plt.subplots()
sns.swarmplot(data=cross_vals_per_cluster)
ax.set_yscale('log')
plt.title('Cross Validation')
plt.xlabel('Cluster')
plt.ylabel('MSE')
plt.savefig(outpath+"cross_validation_log.png")