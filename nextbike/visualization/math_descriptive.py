import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from .. import io


def calculate_aggregate_statistics(p_df_trips):
    """Calculates the following aggregate statistics and saves them as png file:
        - aggr_stats_whole_df: mean and standard deviation of the whole df, of all weekdays and of all weekends
        - calls plot_and_save_aggregate_stats method to do the same on months, days and hours

    Args:
        p_df_trips (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """

    # Split into weekends df and weekdays df
    df_weekends = p_df_trips[p_df_trips['Weekend']]
    df_weekdays = p_df_trips[p_df_trips['Weekend'] == False]

    aggr_stats = {
        "mean_all": p_df_trips['Duration'].mean(),
        "std_all": p_df_trips['Duration'].std(),
        "mean_weekends": df_weekends['Duration'].mean(),
        "std_weekends": df_weekends['Duration'].std(),
        "mean_weekdays": df_weekdays['Duration'].mean(),
        "std_weekdays": df_weekdays['Duration'].std()
    }

    df_aggr_stats = pd.DataFrame.from_dict(aggr_stats, orient="index")
    fig = df_aggr_stats.plot(kind='barh', figsize=(16, 16), fontsize=20).get_figure()
    io.save_fig(p_fig=fig, p_filename='aggr_stats_whole_df.png', p_sub_folder2="math")
    plot_and_save_aggregate_stats(p_df_trips)


def plot_and_save_aggregate_stats(p_df_trips):
    """Aggregates on different time slots.
        - Calculates count, mean and standard deviation
        - Plots them as horizontal bar chart
        - Saves plot as png file

    Args: p_df_trips (DataFrame): Modified DataFrame with trip data from nuremberg (with additional columns month,
    day and hour) Returns: no return
    """

    for time_to_aggregate_on in ["Month_start", "Day_start", "Hour_start"]:
        # data
        x = pd.Series(p_df_trips[time_to_aggregate_on].unique()).sort_values()
        sr_counts = p_df_trips.groupby(by=time_to_aggregate_on)["Duration"].count()
        sr_means = p_df_trips.groupby(by=time_to_aggregate_on)["Duration"].mean()
        sr_stds = p_df_trips.groupby(by=time_to_aggregate_on)["Duration"].std()
        # plotting
        # subplot 1
        width = 0.35
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})
        ax1.bar(x.subtract(width/2), height=sr_means.values, width=width, color="green", label="Mean")
        ax1.bar(x.add(width/2), height=sr_stds.values, width=width, color="red", label="Standard Deviation")
        ax1.set_xticks(x)
        ax1.set_xlabel(time_to_aggregate_on)
        ax1.set_ylabel("Duration [min]")
        ax1.set_title("Mean and Std of Trip Duration per "+time_to_aggregate_on)
        ax1.legend(loc="upper left")
        # subplot 2
        ax2.bar(x, height=sr_counts.values, label="Count")
        if time_to_aggregate_on == "Month_start":
            ax2.set_xticks(x)
        ax2.set_xlabel(time_to_aggregate_on)
        ax2.set_ylabel("Number of Trips")
        ax2.set_title("Count of Trips per "+time_to_aggregate_on)
        io.save_fig(p_fig=fig, p_filename='Aggregate_Statistics_' + time_to_aggregate_on + '.png', p_sub_folder2="math")


def plot_distribution(p_df):
    """Plot the distribution of trip duration including quantile lines.

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """
    # data
    duration = p_df['Duration']
    values, base = np.histogram(duration, bins=int(duration.max()), range=(0, int(duration.max())), weights=np.ones(len(duration)) / len(duration))
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
    io.save_fig(fig, p_filename="DurationMinutes_Distribution.png", p_sub_folder2="math")
    plt.close(fig)


def plot_distribution_monthly(p_df):
    """Plot the distribution of trip lengths per month in violinplots
    and the normal distribution over all months beside.

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """
    # data
    data = p_df[["Duration", "Month_start"]]
    data.loc["Normals"] = None
    months = data["Month_start"].unique()
    for month in months:
        mean = data[data["Month_start"] == month]["Duration"].mean()
        std = data[data["Month_start"] == month]["Duration"].std()
        size = len(data[data["Month_start"] == month])
        normal_distr = np.random.normal(mean, std, size)
        data.loc[data["Month_start"] == month, "Normals"] = normal_distr.astype(np.float64)
    # plotting
    sns.set_style(style="whitegrid")
    #
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), dpi=300, gridspec_kw={'width_ratios': [3, 1]})
    # bw=1 is the scale factor for kernel for nicer visualization
    # cut=0 sets the lower bound of violins to the real lowest duration
    ax1 = sns.violinplot(
        x="Month_start",
        y="Duration",
        data=data,
        ax=ax1,
        bw=1,
        cut=0,
        palette="muted"
    )
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Duration [min]')
    ax1.set_title('Distributions of Durations per Month')
    ax2 = sns.distplot(
        data["Normals"]
    )
    ax2.set_xlim(left=0)
    ax2.set_xlabel('Normalized Duration [min]')
    ax2.set_ylabel('Percentage [%]')
    ax2.set_title("Normal Distribution over all months")
    fig.add_axes(ax1)
    fig.add_axes(ax2)
    io.save_fig(
        fig,
        p_filename="distribution_monthly.png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig)


def corr_analysis(p_df):
    """Plot correlation between features.

    Args:
        p_df (DataFrame): DataFrame of trips in nuremberg
    Returns:
        no return
    """
    corrs = p_df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corrs, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax1 = sns.heatmap(
        corrs,
        mask=mask,
        cmap=cmap,
        center=0,
        ax=ax,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    fig.add_axes(ax1)
    io.save_fig(
        fig,
        p_filename="Correlation.png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig)


def plot_mean_duration(p_df):
    """Plot the mean duration for each day of year and visualize the seasons.

    Args:
        p_df (DataFrame): Dataframe of trips in nuremberg
    Returns:
        no return
    """
    # calculate mean duration of trips for each day of year
    df_day_mean = p_df.groupby(by="Day_of_year_start").mean()[["Duration", "Season"]]
    # create series of days of year
    df_days = pd.DataFrame(columns=["Duration"], index=np.arange(1, 366), data=0)
    # add column mean duration per day to 1 - 365
    df_datapoints = df_days.merge(
        df_day_mean,
        left_on=df_days.index,
        right_on=df_day_mean.index,
        how="left"
    ).set_index("key_0")

    df_datapoints = df_datapoints.fillna(0).rename({"Duration_y": "Duration"}, axis=1)
    df_datapoints = df_datapoints.drop("Duration_x", axis=1)
    x_1 = df_datapoints[df_datapoints["Season"] == 1].index.values
    x_2 = df_datapoints[df_datapoints["Season"] == 2].index.values
    x_3 = df_datapoints[df_datapoints["Season"] == 3].index.values
    x_4 = df_datapoints[df_datapoints["Season"] == 4].index.values
    y_1 = df_datapoints[df_datapoints["Season"] == 1]["Duration"]
    y_2 = df_datapoints[df_datapoints["Season"] == 2]["Duration"]
    y_3 = df_datapoints[df_datapoints["Season"] == 3]["Duration"]
    y_4 = df_datapoints[df_datapoints["Season"] == 4]["Duration"]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Mean Duration of Booking [min]')
    ax.set_title('Mean duration per day')
    ax.bar(x_1, y_1, 1.2, color="cyan", label="Winter")
    ax.bar(x_2, y_2, 1.2, color="red", label="Spring")
    ax.bar(x_3, y_3, 1.2, color="orange", label="Summer")
    ax.bar(x_4, y_4, 1.2, color="green", label="Fall")
    ax.plot(df_datapoints.index, df_datapoints["Duration"], c="black")
    ax.legend(loc='upper right')
    io.save_fig(
        fig,
        p_filename="Mean_Duration_per_Day.png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig)


# TODO: add docstring
def plot_features_influence(p_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    i = 0
    for col in p_df.drop("Duration", axis=1).columns:
        i = i+1
        x = p_df[col]
        y = p_df["Duration"]
        ax.set_xlabel(col)
        ax.set_ylabel("Duration")
        ax.set_title("Duration for each "+col)
        ax.scatter(x, y, s=1, c="blue")
        ax.xaxis.set_ticks(np.arange(min(x), max(x) + 1, max(x) * 0.2))

        io.save_fig(fig, str(i)+col+"_Duration.png", p_sub_folder2="features")
    print("DONE")
