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
    Supposes the excitation functions are all order 2 exponential B-splines.

    :param:
    """
    def __init__(self, M, mu, h) -> None:
        self.M = M  # number of neurons
        self.mu = mu  # vector of M background intensities
        self.h = h  # M x M matrix of activation functions

    def intensity(self, i, t, past_arrivals):
        """Computes, for a fixed i:
        lambda*_i(t) = mu_i + sum n=1 to M, sum t_in<t h_in(t-t_in) for a given multivariate HP

        i: int in {1, ..., M}, corresponding to the process we want to compute the intensity of
        t: time at which to evaluate lambda_i*(t)
        past_arrivals: list containing M arrays of arrival times

        returns lambda*_i(t)
        """
        # Check for errors in input dimensions
        if i < 0 or i > self.M:
            raise ValueError("idx must be an integer in 0, ..., M-1")
        
        if len(past_arrivals) != self.M:
            raise ValueError("List of past arrivals does not contain M arrays.")

        # Compute intensity
        intensity = self.mu[i]
        for n in range(0, self.M):
            intensity += sum(self.h[i, n](t - t_in) for t_in in past_arrivals[n] if t_in < t)

        return intensity

    # def simulate_multivariate_hp(self, T):
    #     """
    #     T: generate process on [0, T]

    #     returns P: array of times of arrivals
    #     """
    #     eps = 10e-10
    #     P = [np.array([]) for _ in range(self.M)]  # list of arrivals
    #     t = np.zeros(self.M)

    #     while np.max(t) < T:
    #         print('started big while loop')
    #         # TODO: is successively looping on each process the best way of doing this ?
    #         # non: avec cette methode, le 2e process ne va jamais generer la 1ere arrivee

    #         # idea: at each time step, loop on all processes, find whichever one generates the first
    #         # arrival, and keep that one as my next arrival, noting the i which generated it (like this
    #         # I can add it to P[i])
    #         temporary_arrivals = np.full((self.M, ), np.inf)
    #         for i in range(0, self.M):
    #             M = self.intensity(i=i, t=t[i]+eps, past_arrivals=P)

    #             r = 1
    #             temp_t = 0
    #             while r >= 1:
    #                 E = np.random.exponential(1/M)
    #                 temp_t = t[i] + E

    #                 U = np.random.uniform(low=0, high=M)
    #                 if temp_t < T and U <= self.intensity(i=i, t=temp_t, past_arrivals=P):
    #                     t[i] = temp_t
    #                     temporary_arrivals[i] = t[i]
    #                     break
    #                 else:
    #                     r += 1
    #                     if r == 1000:
    #                         print('Something went wrong with thinning algo')

    #         print('max value of t is', np.max(t))
    #         # Check which arrival came first and where it came from
    #         first_arrival_idx = np.argmin(temporary_arrivals)
    #         first_arrival = np.min(temporary_arrivals)
    #         # Add that arrival to previous arrivals
    #         #next_arrival = temporary_arrivals[first_arrival_idx]
    #         P[first_arrival_idx] = np.hstack([P[first_arrival_idx], first_arrival])

    #     return P

    def simulate_multivariate_hp(self, T):
        """
        T: generate process on [0, T]

        returns P: array of times of arrivals
        """
        eps = 10e-10
        P = [np.array([]) for _ in range(self.M)]  # list of arrivals
        t = np.zeros(self.M)  # array of candidate arrival times at each time step

        while np.max(t) < T:
            # For each process, determine next arrival
            for j in range(0, self.M):
                # Start from last recorded arrival
                if P[j].size != 0:
                    temp_t = P[j][-1]
                else:
                    temp_t = 0

                # Find next arrival for process j
                r = 0
                while True:
                    M = self.intensity(i=j, t=temp_t+eps, past_arrivals=P)
                    E = np.random.exponential(1/M)
                    temp_t += E

                    U = np.random.uniform(low=0, high=M)
                    if U <= self.intensity(i=j, t=temp_t, past_arrivals=P):
                        t[j] = temp_t
                        break
                    else:
                        r += 1
                        if r==1000:
                            print('sth went wrong with inner loop')
            
            # Once all M candidate arrivals have been generated, pick the first one
            first_arrival_idx = np.argmin(t)
            first_arrival = np.min(t)
            # Add that arrival to previous arrivals
            P[first_arrival_idx] = np.hstack([P[first_arrival_idx], first_arrival])

        return P
