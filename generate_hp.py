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
M = 6  # number of variables/neurons

mu = 0.4*np.ones(M)  # background intensity

sparsity = 2
alpha = np.diag(np.ones(M)*0.7)  # excitation parameters
alpha[np.random.choice(M, size=sparsity, replace=False), 
      np.random.choice(M, size=sparsity, replace=False)] = 0.3

beta = 5

T = 120  # simulation time
print(alpha)

# Warning if HP parameters will make HP explode (TODO: do this with spectral radius)
for (i, j), alpha_ij in np.ndenumerate(alpha):
    if alpha_ij >= beta**2:  # branching ratio = ||h||L1 = alpha/beta^2
        print(f"Need alpha_{i}{j} < beta^2 for the branching ratio to be < 1")

# Define 2nd order B-spline kernels
kernels = np.empty((M, M), dtype=object)

for (i, j), alpha_ij in np.ndenumerate(alpha):
    def hij(t, alpha=alpha_ij, beta=beta):
        return alpha * beta * t * np.exp(-beta*t) * (t >= 0)
    kernels[i, j] = hij

# Define multivariate HP
multihp = MultivariateHP(M=M, mu=mu, h=kernels)

numOfSimulations = 2
for n in range(numOfSimulations):
    print(f'Starting simulation {n+1} out of {numOfSimulations}')
    # Simulate HP
    print('Simulating...')
    P = multihp.simulate_multivariate_hp(T=T)
    print('Finished simulating.')
    for i in range(M):
        print(f"Number of arrivals of process {i}:", len(P[i]))

    # Plot
    # fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    # plot_counting_process(P, M=M, axe=axes[0, 0])
    # plt.show()

    # Make the array rectangular by padding with zeros
    max_len = max(len(row) for row in P) # Find the maximum length of rows in P
    P_array = np.full((len(P), max_len), 0.0) # Init zero array

    for i, row in enumerate(P):
        P_array[i, :len(row)] = row  # Fill the array with the actual values from self.t

    # Write to a csv file
    csv_file_path = f"D:/Users/bassa/Desktop/Code_SemesterProject/simulated_data/my_simulations/simu{n}.csv"  # Change this to your desired path
    # Open the file in write mode with the specified path
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the array to the CSV file
        csv_writer.writerows(P_array)
