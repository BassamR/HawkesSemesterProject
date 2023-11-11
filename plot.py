"""
Module containing functions useful for plotting
"""

import numpy as np
import matplotlib.pyplot as plt

# Plot counting process
def plot_counting_process(P) -> None:
    """
    P: arrival times
    """
    def counting_process(arrival_times):
        return [i + 1 for i in range(len(arrival_times))]

    plt.figure()
    plt.step(x=P, y=counting_process(P), where='post', label='N(t)')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.scatter(P, [0] * len(P), marker='x', color='red', label='Arrival times')
    plt.legend(loc='upper left')

    return

# Plot intensity function
def plot_intensity(P, T, mu, h) -> None:
    """
    P: array of arrivals
    T: arrivals contained in the interval [0, T]
    h: excitation function
    """
    # Function to calculate the sum of f(t - ti)
    def sum_excitations(t, t_values, f):
        return sum(f(t - ti) for ti in t_values if ti < t)

    time_vector = np.linspace(0, T, 500)
    intensity = mu + np.array([sum_excitations(t, P, h) for t in time_vector])

    plt.figure()
    plt.plot(time_vector, intensity, label='$\lambda^*(t)$')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.scatter(P, [0] * len(P), marker='x', color='red', label='Arrival times')
    plt.legend()

    return

# TODO: Plot histogram of arrivals
def plot_histogram(P) -> None:
    return