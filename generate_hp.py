"""
Author: Bassam El Rawas

Script that generates arrivals of a multivariate Hawkes Process, and writes them in a csv file.
"""

import numpy as np
import matplotlib.pyplot as plt
from random import *
import csv

from hawkes import MultivariateHP
from plot_hawkes import *


# Declare HP constants
M = 2  # number of variables/neurons
beta = 5

mu = 0.2*np.ones(M)  # background intensity

sparsity = 2
alpha = np.diag(np.ones(M)*0.4)  # excitation parameters
alpha[np.random.choice(M, size=sparsity, replace=False), 
      np.random.choice(M, size=sparsity, replace=False)] = 0.1

T = 120  # simulation time
print("Excitation matrix:\n", alpha)

# Warning if HP parameters will make HP explode
_, S, _ = np.linalg.svd(alpha)  # stability is determined by matrix H, H_ij = ||hij||_L1
spectralRadius = np.max(S)
if np.max(S) >= 1:
    raise ValueError(f"Spectral radius of excitation matrix is {spectralRadius} > 1, HP is unstable.")
else:
    print(f"Spectral radius is {spectralRadius} < 1, HP is stable.")

# Define 2nd order B-spline kernels
kernels = np.empty((M, M), dtype=object)

for (i, j), alpha_ij in np.ndenumerate(alpha):
    def hij(t):
        return alpha_ij * (beta**2) * t * np.exp(-beta*t) * (t >= 0)
    kernels[i, j] = hij

# Define multivariate HP
multihp = MultivariateHP(M=M, mu=mu, h=kernels)

numOfSimulations = 1
for n in range(numOfSimulations):
    print(f'Starting simulation {n+1} out of {numOfSimulations}')

    # Simulate HP
    print('Simulating...')
    P = multihp.simulate_multivariate_hp(T=T)
    print('Finished simulating.')
    for i in range(M):
        print(f"Number of arrivals of process {i}:", len(P[i]))

    # Plot
    _, ax = plt.subplots(1, 1, figsize= (6,4))
    for i in range(M):
        ax.plot(P[i], np.zeros(len(P[i]))+i, linestyle='', marker='+')
    ax.set_xlabel('Spike trains')
    ax.set_ylabel('Neurons')
    plt.show()

    # Make the array rectangular by padding with zeros
    max_len = max(len(row) for row in P) # Find the maximum length of rows in P
    P_array = np.full((len(P), max_len), 0.0) # Init zero array

    for i, row in enumerate(P):
        P_array[i, :len(row)] = row  # Fill the array with the actual values from self.t

    # Write to a csv file
    # csv_file_path = f"D:/Users/bassa/Desktop/Code_SemesterProject/simulated_data/my_simulations/simu{n}.csv"  # Change this to your desired path
    # # Open the file in write mode with the specified path
    # with open(csv_file_path, 'w', newline='') as csv_file:
    #     # Create a CSV writer object
    #     csv_writer = csv.writer(csv_file)

    #     # Write the array to the CSV file
    #     csv_writer.writerows(P_array)
