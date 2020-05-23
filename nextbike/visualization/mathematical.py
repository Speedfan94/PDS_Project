import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime as dt

from matplotlib.ticker import PercentFormatter

from .. import io


def plot_distribution(df):
    """Plots the distribution of trip lengths per month including quantile lines

    Args:
        df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """
    # Visualize the distribution of trip lengths per month. Compare the distributions to normal
    # distributions with mean and standard deviation as calculated before (1.d))

    # TODO: Code to start on
    # histogram of duration

    # data
    duration = df['Duration']
    values, base = np.histogram(duration, bins=120, range=(0, 120), weights=np.ones(len(duration)) / len(duration))
    quantile_25 = np.quantile(duration, 0.25)
    quantile_50 = np.quantile(duration, 0.5)
    quantile_75 = np.quantile(duration, 0.75)
    quantile_95 = np.quantile(duration, 0.95)

    # plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('Duration of Booking [min]')
    ax.set_ylabel('Percentage')
    ax.set_title('Distribution of Duration')
    plt.plot(base[:-1], values, c='blue')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.vlines(quantile_25, 0, 0.07, linestyles='dashed', label='25% Quantile', colors='green')
    plt.vlines(quantile_50, 0, 0.07, linestyles='dashed', label='50% Quantile', colors='yellow')
    plt.vlines(quantile_75, 0, 0.07, linestyles='dashed', label='75% Quantile', colors='red')
    plt.vlines(quantile_95, 0, 0.07, linestyles='dashed', label='95% Quantile')
    plt.legend(loc='upper right')
    io.save_fig(fig, pFile_name="DurationMinutes_Distribution.png")


def visualize_more(df):
    """TODO: What else can we visualize?

    Args:
        df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """
    # These visualizations are the minimum requirement.
    # Use more visualizations wherever it makes sense.

    print()
