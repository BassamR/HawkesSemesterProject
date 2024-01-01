"""
Author: Bassam El Rawas

Module in which the likelihood operator for a multivariate HP is defined.
Recall that we want to minimize $-\ln(L(\theta)) = E(A\theta)$ where: 
- $E: \mathbb{R}^{M+K} \to \mathbb{R}$, 
  $x \mapsto E(x) = -\sum_{x \not\in S} \ln(x_i) + \sum_{x \in S} x_i$,
  $S \subset \{1, ..., M+K\}$ the set of indices contributing to the compensator term
- $A \in \mathbb{R}^{(M+K) \times (M + M^2)}$ is the discrete likelihood matrix
- $\theta \in \mathbb{R}^{M + M^2}$ is the vector of parameters, 
  $\theta = (\mu_1, \vec{\alpha}_1, ..., \mu_M, \vec{\alpha}_M)$ 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyxu.abc import DiffFunc, LinOp
import pyxu.operator as pxop


# Define E: R^{K+M} -> R, E(x) = -\sum_{i \not\in S} ln(x_i) + \sum_{i \in S} x_i
class LikelihoodE(DiffFunc):
    def __init__(self, dim, S, notS):
        super().__init__(shape=(1, dim))
        self.S = S
        self.notS = notS

    def apply(self, arr):
        return np.array([sum(-np.log(arr[self.notS]+10e-10)) + sum(arr[self.S])])

    def grad(self, arr):
        grad = np.ones(arr.shape)
        grad[self.notS] = -1/(arr[self.notS]+10e-10)
        return grad


class HawkesLikelihood():
    """
    Class that, given a realization, stores parameters relevant to it and defines the necessary
    operators to be able to represent -log(likelihood) as a Pyxu DiffMap.
    """
    def __init__(self, path: str, beta: float) -> None:
        # Initialize constants
        self.t = None  # list of arrival times
        self.T = None  # largest arrival time
        
        self.M = None  # total number of neurons
        self.k = None  # list containing the number of arrivals for each neuron
        self.K = None  # total number of arrivals across all processes

        self.A = None  # all matrices A

        self.opA = None  # discrete likelihood operator
        self.E = None  # convex likelihood operator
        self.negLogL = None  # -log(likelihood)

        # Initialize HP global parameters
        self.beta = beta  # exponential decay constant
        self.betainv = 1/self.beta

        # Read neuron spike data, then convert to list of np arrays
        self.init_constants(path)

        # Compute discrete likelihood matrices A^j, then define them as LinOps
        self.init_matrix_A()
        
        ops = [None for _ in range(self.M)] 
        for i in range(self.M):
            ops[i] = LinOp.from_array(self.A[i]) # Define each A^j as an operator

        # Initialize operator A of shape (K+M, M*(1+M)) as block-diagonal
        self.opA = pxop.block_diag(ops)

        # Define subset S of {1, ..., M+K} of row indices of A which contribute to the compensator term
        S = np.zeros(self.M)  # we know that |S| = M
        for i in range(self.M):
            S[i] = sum(self.k[:(i+1)]) + i + 1
        S = [int(si)-1 for si in S]  # convert to int, decaler de -1 pour avoir tout entre 0 et M+K-1

        # Define the complement of S
        allIndices = list(range(0, self.M + self.K))
        notS = list(set(allIndices) - set(S))

        # Initialize operator E
        dim = self.M + self.K
        self.E = LikelihoodE(dim=dim, S=S, notS=notS)

        # Define -log(likelihood) = E(A*theta) operator (which we want to minimize)
        self.negLogL = self.E * self.opA  # R^{M^2 + M} -> R, via R^{K+M}

        return
    
    def init_constants(self, path: str) -> None:
        """
        Initializes constants based on given realization.
        """
        self.t = pd.read_csv(path, header=None)
        self.t = [self.t.iloc[i].values for i in range(len(self.t))]

        # Obtain useful constants
        self.M = len(self.t)  # number of neurons

        for i in range(self.M):
            self.t[i] = self.t[i][np.nonzero(self.t[i])]  # get rid of arrival times = 0

        self.k = [len(ti) for ti in self.t] 
        self.T = max(max(self.t[i]) for i in range(self.M)) + 1e-8
        self.K = sum(self.k)

        return

    def init_matrix_A(self) -> None:
        """
        Computes the discrete likelihood matrix A.
        """
        # Shorthands
        betainv = self.betainv
        beta = self.beta
        g = self.g

        # Turn self.t into a numpy array by padding with NaNs
        max_len = max(len(row) for row in self.t) # Find the maximum length of rows in self.t
        t_array = np.full((len(self.t), max_len), np.nan) # Init NaN array

        for i, row in enumerate(self.t):
            t_array[i, :len(row)] = row  # Fill the array with the actual values from self.t

        # Initialize matrix
        self.A = [np.zeros((self.k[j]+1, self.M+1)) for j in range(self.M)]

        # Compute last row of every A^j
        B = np.zeros(self.M)
        for n in range(self.M):
            B[n] = betainv * sum((self.T - tnl + betainv)*np.exp(-beta*(self.T - tnl)) - betainv for tnl in self.t[n])

        # Fill out A^j for every j
        for j in range(self.M):
            print(f'Computing A^{j+1} out of {self.M}, k{j+1}={self.k[j]}')
            # First column computation
            self.A[j][:, 0] = 1
            self.A[j][-1, 0] = -self.T

            # Fill out last row
            self.A[j][-1, 1:] = B

            # Compute the rest of the matrix row by row
            for i in range(self.A[j].shape[0] - 1):
                # Create a mask for values more than self.t[j][i]
                mask = t_array >= self.t[j][i]
                diff = self.t[j][i] - t_array
                diff[mask] = -1  # g is an exponential kernel, do this to avoid overflow
                filtered_values = g(diff)
                
                result = np.nansum(filtered_values, axis=1)
                self.A[j][i, 1:] = result[:self.M]

        return

    def plot_realization(self) -> None:
        """
        Plots the spike trains of all neurons.
        """
        _, ax = plt.subplots(1, 1, figsize= (10,8))
        for i in range(self.M):
            ax.plot(self.t[i], np.zeros(len(self.t[i]))+i, linestyle='', marker='+')
        ax.set_xlabel('Spike trains')
        ax.set_ylabel('Neurons')
        plt.show()
        return

    def plot_A(self, idx: int) -> None:
        """
        Plots a heatmap of the matrix A[idx], without the last row as to keep things
        to scale.
        """
        fig, ax = plt.subplots(1, 1, figsize=(5,8))
        im = ax.imshow(self.A[idx][0:-1, :], 
                       cmap='viridis', 
                       interpolation='nearest', 
                       extent=[1, self.M, 0, self.k[idx]/5])
        fig.colorbar(im, ax=ax)  # Add a colorbar to show the scale
        ax.set_title('Matrix Coefficients')
        plt.show()
        return

    def g(self, t):
        # Normalized causal Green's function of L = (D + beta*I)^2
        return (self.beta ** 2) * t * np.exp(-self.beta*t) * (t >= 0)
