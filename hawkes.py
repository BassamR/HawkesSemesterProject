"""
Module containing functions to simulate and plot a univariate Hawkes Process.
"""

import numpy as np
import matplotlib.pyplot as plt

class UnivariateHP:
    """Class used to simulate realizations of a univariate HP. Stores its parameters.
    
    :param mu: Background intensity.
    :param h: Excitation function.
    """
    def __init__(self, mu, h):
        self.mu = mu  # background intensity
        self.h = h  # excitation function

    def intensity(self, t, past_arrivals) -> float:
        """Computes lambda*(t) = mu + sum h(t-ti) for ti<t for a given univariate HP

        t: time at which to evaluate lambda*(t)
        past_arrivals: array containing past arrival times

        returns lambda*(t)
        """
        return self.mu + sum(self.h(t - ti) for ti in past_arrivals if ti < t)
    
    # Thinning algorithm to generate arrivals
    def simulate_univariate_hp(self, T) -> np.array:
        """
        T: generate process on [0, T]

        returns P: array of times of arrivals
        """
        eps = 10e-10
        P = np.array([])  # array of arrivals
        t = 0

        while t < T:
            M = self.intensity(t=t+eps, past_arrivals=P)
            E = np.random.exponential(1/M)
            t = t + E

            U = np.random.uniform(low=0, high=M)
            if t < T and U <= self.intensity(t=t, past_arrivals=P):
                P = np.hstack([P, t])

        return P

class MultivariateHP:
    """Class used to simulate realizations of a multivariate HP. Stores its parameters.

    :param:
    """
    def __init__(self) -> None:
        pass
