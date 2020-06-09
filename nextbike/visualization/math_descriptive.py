import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from nextbike import io

# define color constants
COLOR_BAR_MEAN = "yellowgreen"
COLOR_BAR_STD = "firebrick"
FONTSIZE_TITLE = 18
FONTSIZE_AXIS_LABEL = 16


def calculate_aggregate_statistics(p_df_trips, p_mode):
    """Calculates the following aggregate statistics and saves them as png file:
        - aggr_stats_whole_df: mean and standard deviation of the whole df, of all weekdays and of all weekends
        - calls plot_and_save_aggregate_stats method to do the same on months, days and hours

    Args:
        p_df_trips (DataFrame): DataFrame with trip data from nuremberg
        p_mode (str): describes training or testing parameter
    Returns:
        No return
    """
    # get first and last date in data to print out period of time
    first_date, last_date = p_df_trips["Start_Time"].agg(["min", "max"]).dt.strftime("%d.%m.%y")

    # plot other subsets
    timeperiods_to_aggregate_on = [
        "Weekend",
        "Day_of_week_start",
        "Month_start",
        "Day_start",
        "Hour_start",
        "Day_of_year_start",
        "Season"
    ]
    for timeperiod in timeperiods_to_aggregate_on:
        subsets = p_df_trips[timeperiod].unique()
        subset_labels = pd.Series(subsets).sort_values().values
        aggr_stats = {
            "subset": subset_labels,
            "mean": p_df_trips.groupby(by=timeperiod)["Duration"].mean(),
            "std": p_df_trips.groupby(by=timeperiod)["Duration"].std(),
            "count": p_df_trips.groupby(by=timeperiod)["Duration"].count()
        }
        df_aggr_stats = pd.DataFrame.from_dict(aggr_stats)
        total_stats = {
            "subset": "total",
            "mean": p_df_trips["Duration"].mean(),
            "std": p_df_trips["Duration"].std(),
            "count": p_df_trips["Duration"].count()
        }

        plot_and_save_aggregate_stats(
            p_df_aggr_stats=df_aggr_stats,
            p_total_stats=total_stats,
            p_aggr_time_period=timeperiod,
            p_first_date=first_date,
            p_last_date=last_date,
            p_mode=p_mode
        )


def plot_and_save_aggregate_stats(p_df_aggr_stats, p_total_stats, p_aggr_time_period, p_first_date, p_last_date, p_mode):
    """Aggregates on different time slots.
        - Calculates count, mean and standard deviation
        - Plots them as horizontal bar chart
        - Saves plot as png file

    Args:
        p_df_aggr_stats (DataFrame): aggregation statistics (mean, std, count) for each subset
        p_total_stats: statistics of whole dataframe (mean, std, count)
        p_aggr_time_period: time period to aggregate on (aka column name)
        p_first_date: first date of dataset
        p_last_date: last date of dataset
    Returns:
        No return
    """
    # set general plotting params
    counts_total = p_total_stats["count"]
    p_df_aggr_stats["count_percentage"] = p_df_aggr_stats.apply(lambda row: row["count"]/counts_total*100, axis=1)
    p_df_aggr_stats["label_pie_legend"] = p_df_aggr_stats.apply(lambda row: build_pie_legend_label(row), axis=1)

    # create colors for pie chart
    colormap = cm.Spectral(np.linspace(0, 1, len(p_df_aggr_stats)))
    # alternative colormaps
    # colormap = cm.hsv(np.linspace(0, 1, len(subset_labels)))
    # colormap = cm.viridis(np.linspace(0, 1, len(subset_labels)))
    colors_pie_counts = [[0, 0, 0, 0.6]] + [list(color[:3]) + [0.8] for color in colormap]

    labels_pie_title = "Count of Trips\n"\
                       "Total number of Trips: " + str(counts_total) + "\n"\
                       "(From " + p_first_date + " to " + p_last_date + ")"

    df_aggr_stats_bar_chart = p_df_aggr_stats.append(p_total_stats, ignore_index=True)

    # plotting
    width = 0.35
    # generate x values
    all_x_values = p_df_aggr_stats["subset"].replace({True: 1, False: 0})
    x_min, x_max = all_x_values.agg(["min", "max"])
    # add x value for total bar
    x_total = x_max+2
    x = np.append(all_x_values.values, x_total)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8), dpi=300, gridspec_kw={"width_ratios": [2, 1]})
    # subplot 1
    ax1.bar(
        x-(width/2),
        height=df_aggr_stats_bar_chart["mean"].values,
        width=width,
        color=COLOR_BAR_MEAN,
        label="Mean"
    )
    ax1.bar(
        x+(width/2),
        height=df_aggr_stats_bar_chart["std"].values,
        width=width,
        color=COLOR_BAR_STD,
        label="Standard Deviation"
    )
    # set x ticklabels depending on the number of x ticks
    subset_labels = df_aggr_stats_bar_chart["subset"]
    if x_total > 360:
        # day of year: only show ticklabel every 30 ticks
        labels = np.arange(x_min, x_max + 1, 30)
        x_ticks = np.append(labels, x_total)
        x_ticklabels = np.append(labels, "total")
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticklabels, rotation=90)
    elif x_total > 25:
        # day of month: only show ticklabel every 5 ticks
        labels = np.arange(x_min, x_max+1, 5)
        x_ticks = np.append(labels, x_total)
        x_ticklabels = np.append(labels, "total")
        ax1.set(xticks=x_ticks, xticklabels=x_ticklabels)
    else:
        # show every ticklabel
        ax1.set_xticks(x)
        ax1.set_xticklabels(subset_labels)

    ax1.set_xlabel(p_aggr_time_period, fontsize=FONTSIZE_AXIS_LABEL)
    ax1.set_ylabel("Duration [min]", fontsize=FONTSIZE_AXIS_LABEL)
    ax1.set_title("Mean and Std of Trip Duration ("+p_aggr_time_period+")", fontsize=FONTSIZE_TITLE)
    ax1.legend(loc="upper left")

    # subplot 2 (pie chart: weekend vs weekday)
    patches, texts = ax2.pie(p_df_aggr_stats["count_percentage"],
                             colors=colors_pie_counts,
                             startangle=90,
                             counterclock=False,
                             wedgeprops={"linewidth": 0.5, "edgecolor": "dimgrey"})

    ax2.axis("equal")
    ax2.set_ylim(bottom=-3, top=1.5)
    ax2.set_title(labels_pie_title, fontsize=FONTSIZE_TITLE)
    patches_legend = patches
    labels_legend = p_df_aggr_stats["label_pie_legend"]
    pie_legend_title = p_aggr_time_period
    if len(labels_legend) > 32:
        # too many elements in legends, only show top 30
        df_aggr_stats = p_df_aggr_stats.reset_index(drop=True)
        df_top_entries = df_aggr_stats.sort_values(by=["count"], ascending=False).head(30)
        labels_legend = df_top_entries["label_pie_legend"].values
        patches_legend = []
        colors_legend = []
        pie_legend_title = pie_legend_title+" (Top 30)"
        for top_entry_index in df_top_entries.index:
            # TODO: Legend colors are broken after filtering legend items
            colors_legend.append(colors_pie_counts[top_entry_index])
            patches_legend.append(patches[top_entry_index])
    ax2.legend(patches_legend,
               labels=labels_legend,
               loc="center",
               bbox_to_anchor=(0.5, 0.2),
               ncol=2,
               title=pie_legend_title)

    io.save_fig(
        p_fig=fig,
        p_filename="Aggregate_Statistics_"+p_aggr_time_period+p_mode+".png",
        p_sub_folder2="math"
    )
    plt.close()


def build_pie_legend_label(p_row):
    """Builds label for pie legend entry
    (entry name, count percentage and absolute count)

    Args:
        p_row: current row (wedge of the pie chart)
    Returns:
        label
    """
    if type(p_row["subset"]) == bool:
        str_subset_name = str(p_row["subset"])
    else:
        # cut off decimals before turning into string
        str_subset_name = str(int(p_row["subset"]))
    # round to two decimals before turning into string
    str_count_percentage = str(np.round(p_row["count_percentage"], 2))
    str_count_absolute = str(int(p_row["count"]))
    return str_subset_name+": "+str_count_percentage+"% ("+str_count_absolute+" trips)"


def plot_distribution(p_df, p_mode):
    """Plot the distribution of trip duration including quantile lines.

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        No return
    """
    # data
    duration = p_df["Duration"]
    values, base = np.histogram(duration, bins=int(duration.max()), range=(0, int(duration.max())), weights=np.ones(len(duration)) / len(duration))
    quantile_25 = np.quantile(duration, 0.25)
    quantile_50 = np.quantile(duration, 0.5)
    quantile_75 = np.quantile(duration, 0.75)
    quantile_95 = np.quantile(duration, 0.95)
    # plotting
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.set_xlabel("Duration of Booking [min]", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylabel("Percentage", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_title("Distribution of Duration", fontsize=FONTSIZE_TITLE)
    plt.plot(base[:-1], values, c="blue")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.vlines(quantile_25, 0, 0.07, linestyles="dashed", label="25% Quantile", colors="green")
    plt.vlines(quantile_50, 0, 0.07, linestyles="dashed", label="50% Quantile", colors="yellow")
    plt.vlines(quantile_75, 0, 0.07, linestyles="dashed", label="75% Quantile", colors="red")
    plt.vlines(quantile_95, 0, 0.07, linestyles="dashed", label="95% Quantile")
    plt.legend(loc="upper right")
    io.save_fig(fig, p_filename="DurationMinutes_Distribution"+p_mode+".png", p_sub_folder2="math")
    plt.close(fig)


def plot_distribution_monthly(p_df, p_mode):
    """Plot the distribution of trip lengths per month in violinplots
    and the normal distribution over all months beside.

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        No return
    """
    # data
    # important: take a copy of p_df columns!
    # otherwise pandas throws warnings when initializing column "Normals" with None values
    data = p_df[["Duration", "Month_start"]].copy()
    data.loc[:, "Normals"] = None
    months = data["Month_start"].unique()
    for month in months:
        mean = data[data["Month_start"] == month]["Duration"].mean()
        std = data[data["Month_start"] == month]["Duration"].std()
        size = len(data[data["Month_start"] == month])
        normal_distr = np.random.normal(mean, std, size)
        data.loc[data["Month_start"] == month, "Normals"] = normal_distr.astype(np.float64)

    # plotting
    sns.set_style(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8), dpi=300, gridspec_kw={"width_ratios": [3, 1]})
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
    ax1.set_xlabel("Month", fontsize=FONTSIZE_AXIS_LABEL)
    ax1.set_ylabel("Duration [min]", fontsize=FONTSIZE_AXIS_LABEL)
    ax1.set_title("Distributions of Durations per Month", fontsize=FONTSIZE_TITLE)
    ax2 = sns.distplot(
        data["Normals"]
    )
    ax2.set_xlim(left=0)
    ax2.set_xlabel("Normalized Duration [min]", fontsize=FONTSIZE_AXIS_LABEL)
    ax2.set_ylabel("Percentage [%]", fontsize=FONTSIZE_AXIS_LABEL)
    ax2.set_title("Normal Distribution over all months", fontsize=FONTSIZE_TITLE)
    fig.add_axes(ax1)
    fig.add_axes(ax2)
    io.save_fig(
        fig,
        p_filename="distribution_monthly"+p_mode+".png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig)


def corr_analysis(p_df, p_weather):
    """Plot correlation between features.

    Args:
        p_df (DataFrame): DataFrame of trips in nuremberg
    Returns:
        No return
    """
    corrs = p_df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corrs, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

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
        p_filename="Correlation"+p_weather+".png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig)


def plot_mean_duration(p_df, p_mode):
    """Plot the mean duration for each day of year and visualize the seasons.

    Args:
        p_df (DataFrame): Dataframe of trips in nuremberg
    Returns:
        No return
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
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.set_xlabel("Day of year", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylabel("Mean Duration of Booking [min]", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_title("Mean duration per day", fontsize=FONTSIZE_TITLE)
    ax.bar(x_1, y_1, 1.2, color="cyan", label="Winter")
    ax.bar(x_2, y_2, 1.2, color="red", label="Spring")
    ax.bar(x_3, y_3, 1.2, color="orange", label="Summer")
    ax.bar(x_4, y_4, 1.2, color="green", label="Fall")
    ax.plot(df_datapoints.index, df_datapoints["Duration"], c="black")
    ax.legend(loc="upper right")
    io.save_fig(
        fig,
        p_filename="Mean_Duration_per_Day"+p_mode+".png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig)


def plot_pca_components(p_pca_explained_var, p_filename):
    """Plots the PCA components and their explained variance by component.

    Args:
        p_pca_explained_var:    explained variance by component
        p_filename:             filename to save plot as
    Returns:
        No return
    """
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.set_xlabel("Component", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylabel("Explained Variance by Component", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_title("Explained Variance by Principal Component (sum = "+str(sum(p_pca_explained_var))+")",
                 fontsize=FONTSIZE_TITLE)
    ax.bar(np.arange(len(p_pca_explained_var)), p_pca_explained_var)
    io.save_fig(
        fig,
        p_filename=p_filename+".png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig)


def plot_features_influence(p_df):
    """Plots the influence of each feature on the duration

    Args:
        p_df:   whole data set
    Returns:
        No return
    """
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    i = 0
    for col in p_df.drop("Duration", axis=1).columns:
        i = i+1
        x = p_df[col]
        y = p_df["Duration"]
        ax.set_xlabel(col, fontsize=FONTSIZE_AXIS_LABEL)
        ax.set_ylabel("Duration", fontsize=FONTSIZE_AXIS_LABEL)
        ax.set_title("Duration for each "+col, fontsize=FONTSIZE_TITLE)
        ax.scatter(x, y, s=1, c="blue")
        ax.xaxis.set_ticks(np.arange(min(x), max(x) + 1, max(x) * 0.2))

        io.save_fig(fig, str(i)+col+"_Duration.png", p_sub_folder2="features")
        plt.close()
    print("DONE")
