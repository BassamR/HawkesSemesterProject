"""
Module containing functions useful for plotting
"""

import numpy as np
import matplotlib.pyplot as plt

# Plot counting process
def plot_counting_process(P=None, M=1, axe=None) -> None:
    """If M=1, plots N(t) and the arrival times, if M>1 plots all Ni(t) and all the arrival times.

    P: Arrival times, either an array (when M=1) or a list of arrays (when M>1).
    M: Dimension of the HP
    axe: Axe for plotting
    """
    if M < 1:
        raise ValueError("Dimension of the HP must be >=1")
    
    def counting_process(arrival_times):
        return [i + 1 for i in range(len(arrival_times))]
    
    # Realistically I won't be plotting many arrivals for M>3
    colors = ['red', 'green', 'blue']

    if M > 1:
        for i in range(0, len(P)):
            axe.step(x=P[i], y=counting_process(P[i]), where='post', label=f'$N_{i}(t)$')
            axe.scatter(P[i], [0] * len(P[i]), marker='x', color=colors[i], label=f'Arrival times of $N_{i}$')
    else:
        axe.step(x=P, y=counting_process(P), where='post', label='$N(t)$')
        axe.scatter(P, [0] * len(P), marker='x', color='red', label='Arrival times of $N$')

    axe.set_xlabel('Time')
    axe.set_ylabel('Count')
    axe.legend(loc='upper left')
    axe.grid()

    return

# Plot intensity function
def plot_intensity(P, T, mu, h, axe) -> None:
    """
    P: array of arrivals
    T: arrivals contained in the interval [0, T]
    mu: background intensity
    h: excitation function
    axe: Axe for plotting
    """
    # Function to calculate the sum of f(t - ti)
    def sum_excitations(t, t_values, f):
        return sum(f(t - ti) for ti in t_values if ti < t)

    time_vector = np.linspace(0, T, 500)
    intensity = mu + np.array([sum_excitations(t, P, h) for t in time_vector])

    # TODO: check if this expectation computation is ok
    # expectation = np.zeros_like(time_vector)
    # for i in range(len(time_vector)):
    #     expectation[i] = np.mean(intensity[:(i+1)])

    axe.plot(time_vector, intensity, label='$\lambda^*(t)$', linewidth=0.8)
    #axe.plot(time_vector, expectation)
    axe.grid()
    axe.set_xlabel('Time')
    axe.set_ylabel('Intensity')
    axe.scatter(P, [0] * len(P), marker='x', color='red', label='Arrival times')
    axe.legend()

    return

def plot_intensity_multivariate(P, i, T, mu, h, axe) -> None:
    """
    P: list containing M array of arrivals
    i: index of the process whose intensity we want to plot
    T: arrivals contained in the interval [0, T]
    h: excitation functions
    axe: Axe for plotting
    """
    # Function to calculate the sum of f(t - ti)
    def sum_excitations(t, t_values, f):
        return sum(f(t - ti) for ti in t_values if ti < t)
    
    M = len(P)  # total number of processes
    time_vector = np.linspace(0, T, 500)

    # Calculate intensity
    intensity = mu[i]
    for n in range(0, M):
        intensity += np.array([sum_excitations(t_in, P[n], h[i,n]) for t_in in time_vector])

    # Realistically I won't be plotting many arrivals for M>3
    colors = ['red', 'green', 'blue']

    # Plot
    axe.plot(time_vector, intensity, label=f'$\lambda^*_{i}(t)$', linewidth=0.8)
    axe.grid()
    axe.set_xlabel('Time')
    axe.set_ylabel('Intensity')
    for n in range(0, M):
        # Plot arrival times
        axe.scatter(P[n], [0] * len(P[n]), marker='x', color=colors[n], label=f'Arrival times of $N_{n}$')
    axe.legend()

    return

# Plot histogram of arrivals
def plot_histogram(P, T, axe) -> None:
    """
    :param P: Array of arrival times.
    :param T: Arrival times are contained in [0,T].
    :param axe: Axe for plotting.
    """

    axe.hist(P, bins=T//5, edgecolor='black')
    axe.set_xlabel('Arrival Times')
    axe.set_ylabel('Frequency')
    #axe.set_title('Histogram of Arrival Times')

    return
