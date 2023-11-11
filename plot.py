"""
Module containing functions useful for plotting
"""

import numpy as np
import matplotlib.pyplot as plt

# Plot counting process
def plot_counting_process(P, axe) -> None:
    """
    P: arrival times
    axe: Axe for plotting
    """
    def counting_process(arrival_times):
        return [i + 1 for i in range(len(arrival_times))]

    axe.step(x=P, y=counting_process(P), where='post', label='N(t)')
    axe.grid()
    axe.set_xlabel('Time')
    axe.set_ylabel('Count')
    axe.scatter(P, [0] * len(P), marker='x', color='red', label='Arrival times')
    axe.legend(loc='upper left')

    return

# Plot intensity function
def plot_intensity(P, T, mu, h, axe) -> None:
    """
    P: array of arrivals
    T: arrivals contained in the interval [0, T]
    h: excitation function
    axe: Axe for plotting
    """
    # Function to calculate the sum of f(t - ti)
    def sum_excitations(t, t_values, f):
        return sum(f(t - ti) for ti in t_values if ti < t)

    time_vector = np.linspace(0, T, 500)
    intensity = mu + np.array([sum_excitations(t, P, h) for t in time_vector])

    axe.plot(time_vector, intensity, label='$\lambda^*(t)$', linewidth=0.8)
    axe.grid()
    axe.set_xlabel('Time')
    axe.set_ylabel('Intensity')
    axe.scatter(P, [0] * len(P), marker='x', color='red', label='Arrival times')
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
