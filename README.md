# DeepAtlas

Welcome to DeepAtlas: a tool for effective manifold learning.

DeepAtlas is currently a Python-based command-line tool. Please follow the steps below to run:

## 0: Generate Shape Data

This step is optional, for creating data to explore if needed. If you have a dataset of interest already, feel free to skip to Step 1.

You will be prompted to enter a number of points and a shape. The shape options are: hypersphere, s_curve, swiss_roll

Input: Filepath to save the data to
Output: Saves data file(s) to the specified location

## 1: Local Neighborhoods

This step applies k-means clustering to separate the data into local neighborhoods.

You will be prompted to enter a data nickname for file-saving purposes. You will also be asked to choose an integer value of k for k-means clustering. When choosing k, we recommend ensuring each cluster has at least 50 points in it.

Input:
1. Filepath to the high dimensional data in .csv with no row names or column names
2. Filepath to save the result to
Output: Creates a folder named "clusters" and saves k-means clusters of the data

## 2: PC vs. AJD

This step calculates the Average Jaccard Distance (AJD) at each PC. Once the result is plotted, please examine to determine whether a lower-dimensional manifold exists and at what dimension if so. If there is a lower-dimensional manifold, we expect to see the lines for each cluster exhibiting similar behavior and with a low AJD at the manifold dimension.

Input:
1. Filepath to k-means clusters. If you have not made changes, this will be the path to the directory "clusters" made in Step 1
2. Filepath to save the result to
Output: Saves the AJD vs. PC data and displays a plot for visual inspection

## 3: Neural Network

This step trains a neural network to yield a model that maps between the high dimensional space and the PCA embedding space in both directions.

You will be prompted for the following information to design the neural network:
- number of training epochs
- number of layers in the neural network
- PC embedding dimension
- activation function
- number of folds for cross-validation

By default, the neural network will be fully connected and each layer will have the same number of nodes (equal to the high dimension). In order for the neural network to be invertible, the PCA embedding is padded with 0s and we recommend using the ajd_tanh activation function. This activation function is an adjusted version of tanh such that the inverse will always be defined.

Input:
1. Filepath to k-means clusters. If you have not made changes, this will be the path to the directory "clusters" made in Step 1
2. Filepath to save the result to
Output: Saves neural network training information including:
- plots of loss over time during training
- a log of the loss at each training epoch
- models at a regular epoch interval (by default, every 50 epochs)
- the best model for each cluster, as determined by Mean Squared Error (MSE) with the PCA embedding (saved to "models")
- a violin plot of AJD comparisons between the neural network output and the original data, between the PCA embedding and the original data, and between the neural network output and the PCA embedding
- a plot of the MSE across cross-validation folds

## 4: Invert

This step allows the user to apply their neural network model in the inverse direction. Note that this code is currently set up for the default scenario of a model that used the adj_tanh activation and embedded with PCA from 3D into 2D.

You will be prompted for a value n which represents the nth nearest neighbor, to be used as a radius for sampling new points from a sphere around existing points in the lower dimensional manifold space.

Input:
1. Filepath to k-means clusters. If you have not made changes, this will be the path to the directory "clusters" made in Step 1
2. Filepath to the transition json made in Step 1. If you have not made changes, this json will be found in the outpath location passed in to Step 1
3. Filepath to neural network models to use for inversion. If you have not made changes, this will be the path to the directory "models" made in Step 3

Output: Displays the 3D plot resulting from passing the sampled data through the model in the inverse direction and the 3D plot of the direct inverse (no sampling)