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

# Global HP parameters
beta = 1  # we choose this
betainv = 1/beta
def g(t, beta=beta):
    return t * np.exp(-beta*t) * (t >= 0)

# Define E: R^{K+M} -> R, E(x) = -\sum_{i \not\in S} ln(x_i) + \sum_{i \in S} x_i
class LikelihoodE(DiffFunc):
    def __init__(self, dim, S, notS):
        super().__init__(shape=(1, dim))
        self.S = S
        self.notS = notS

    def apply(self, arr):
        return sum(-np.log(arr[self.notS])) + sum(arr[self.S])

    def grad(self, arr):
        grad = np.ones(arr.shape)
        grad[self.notS] = -1/arr[self.notS]
        return grad


class HawkesLikelihood():
    """
    Class that, given a realization, stores parameters relevant to it and defines the necessary
    operators to be able to represent -log(likelihood) as a Pyxu DiffMap.
    """
    def __init__(self, path: str) -> None:
        # Initialize constants
        self.data = None  # list of arrival times
        self.t = None  # same as data, but for easier notation
        self.T = None  # largest arrival time
        
        self.M = None  # total number of neurons
        self.k = None  # list containing the number of arrivals for each neuron
        self.K = None  # total number of arrivals across all processes

        self.A = None  # all matrices A

        self.opA = None  # discrete likelihood operator
        self.E = None  # convex likelihood operator
        self.negLogL = None  # -log(likelihood)

        # Read neuron spike data, then convert to list of np arrays
        self.init_constants(path)

        # Compute discrete likelihood matrices A^j, then define them as LinOps
        self.init_matrix_A()
        
        ops = [None for _ in range(self.M)] 
        for i in range(self.M):
            ops[i] = LinOp.from_array(self.A[i]) # Define each A^j as an operator

        # Initialize operator A of shape (K+M, M*(1+M)) as block-diagonal
        self.opA = pxop.block_diag(ops)  # TODO: maybe implement this in a matrix free way ?

        # Define subset S of {1, ..., M+K} of row indices of A which contribute to the compensator term
        S = np.zeros(self.M)  # we know that |S| = M
        for i in range(self.M):
            S[i] = sum(self.k[:(i+1)]) + i + 1
        S = [int(si)-1 for si in S]  # convert to int, decaler de -1 pour avoir tout entre 0 et M+K-1

        # Define the complement of S
        allIndices = list(range(0, self.M + self.K))  # or range(1, M+K+1)
        notS = list(set(allIndices) - set(S))

        # Initialize operator E
        dim = self.M + self.K
        self.E = LikelihoodE(dim=dim, S=S, notS=notS)

        # Define -log(likelihood) = E(A*theta) operator (which we want to minimize)
        self.negLogL = self.E * self.opA

        return
    
    def init_constants(self, path: str) -> None:
        """
        Initializes constants based on given realization.
        """
        self.data = pd.read_csv(path, header=None)
        self.data = [self.data.iloc[i].values for i in range(len(self.data))]

        # Obtain useful constants
        self.M = len(self.data)  # number of neurons

        self.t = self.data  # pour faciliter les notations apres
        for i in range(self.M):
            self.t[i] = self.t[i][np.nonzero(self.t[i])]  # get rid of arrival times = 0

        self.k = [len(ti) for ti in self.t] 
        self.T = max(max(self.t[i]) for i in range(self.M))  # TODO: Do i need T to be the last arrival, or > last arrival ?
        self.K = sum(self.k)

        return

    def init_matrix_A(self) -> None:
        """
        Computes the discrete likelihood matrix A.
        """
        self.A = [np.zeros((self.k[j]+1, self.M+1)) for j in range(self.M)]

        B = np.zeros(self.M)  # to help compute the last row of A^j
        for n in range(self.M):
            B[n] = betainv * sum((self.T - tnl + betainv)*np.exp(-beta*(self.T - tnl)) - betainv for tnl in self.t[n])

        # Fill out A^j for every j
        for j in range(self.M):
            # First column computation
            self.A[j][:, 0] = 1
            self.A[j][-1, 0] = -self.T

            # Fill out the rest of the matrix (-1 because last row is filled out separately)
            for i in range(self.A[j].shape[0] - 1):
                for n in range(self.M):
                    self.A[j][i, n+1] = sum(g(self.t[j][i] - tnl) for tnl in self.t[n] if tnl < self.t[j][i])

            # Fill out last row
            for n in range(self.M):
                self.A[j][-1, n+1] = B[n]

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