import sys
import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_s_curve

outpath = sys.argv[1] # Path to save result to

shape = input("What shape would you like to generate data for? Your options are: hypersphere, s_curve, swiss_roll: ")
num_points = int(input("How many points should make up the shape? Please enter an integer: "))

def hypersphere(n_dimensions,n_samples=1000,k_space=100,section=False,offset=0, offset_dimension=0,noise=False,noise_amplitude=.01):
    # Function to generate a hypersphere
    random.seed(1)
    data = np.zeros((n_samples,k_space))
    i = 0
    while i < n_samples:
        j = 0
        while j < n_dimensions:
            if section == True:
                a = random.random()
            else:
                a = np.random.normal(0,1)
            data[i,j]=a
            j += 1
        norm = np.linalg.norm(data[i])
        if noise == False:
            data[i] = data[i]/norm
        if noise == True:
            noise_term = (random.uniform(-1,1) * noise_amplitude)
            data[i] = (data[i]/norm) + noise_term
        i += 1
    j = offset_dimension
    if offset != 0:
        i = 0
        while i < n_samples:
            data[i,j] = offset
            i += 1
    data = pd.DataFrame(data)
    return data

def write_data_csv(shape, num_points):
    # Function to write data of the following test shapes: hypersphere, s_curve, swiss_roll
    if shape == 'hypersphere':
        dims = input("How many dimensions of a hypersphere would you like? ")
        space = input("How many dimensions would you like the hypersphere embedded in? ")
        data = hypersphere(n_dimensions=int(dims), n_samples=int(num_points), k_space=int(space))
        filename = str(dims)+"D_sphere_in_"+str(space)+"D_space_"+str(num_points) + "_points.csv"
        data.to_csv(outpath+filename, index=False, header=False)
    else:
        if shape == 's_curve':
            data = make_s_curve(num_points)
        elif shape == 'swiss_roll':
            data = make_swiss_roll(num_points)
        else:
            print("Please try again. Your options are: hypersphere, s_curve, swiss_roll.")
            return
        data_3d = pd.DataFrame(data[0])
        data_3d = pd.DataFrame(data[0])
        data_2d = pd.DataFrame.from_dict({0:data[0][:,1], 1:data[1]})
        
        filename = shape+"_"+str(num_points) + "points.csv"
        data_3d.to_csv(outpath+"3d_"+filename, index=False, header=False)
        data_2d.to_csv(outpath+"2d_"+filename, index=False, header=False)
    print("Data saved.")

write_data_csv(shape, num_points)