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

    for time_to_aggregate_on in ["month_start", "day_start", "hour_start"]:
        sr_counts = p_df_trips[["Duration", time_to_aggregate_on]].groupby(by=time_to_aggregate_on).count()
        fig = sr_counts.plot(kind='barh', figsize=(16, 16), fontsize=22).get_figure()
        io.save_fig(p_fig=fig, p_filename='counts_' + time_to_aggregate_on + '.png', p_sub_folder2="math")
        sr_means = p_df_trips[["Duration", time_to_aggregate_on]].groupby(by=time_to_aggregate_on).mean()
        fig = sr_means.plot(kind='barh', figsize=(16, 16), fontsize=22).get_figure()
        io.save_fig(p_fig=fig, p_filename='means_' + time_to_aggregate_on + '.png', p_sub_folder2="math")
        sr_stds = p_df_trips[["Duration", time_to_aggregate_on]].groupby(by=time_to_aggregate_on).std()
        fig = sr_stds.plot(kind='barh', figsize=(16, 16), fontsize=22).get_figure()
        io.save_fig(p_fig=fig, p_filename='stds_' + time_to_aggregate_on + '.png', p_sub_folder2="math")


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


# TODO
def plot_distribution_monthly(p_df):
    """Plot the distribution of trip lengths per month including quantile lines

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """
    # Visualize the distribution of trip lengths per month. Compare the distributions to normal
    # distributions with mean and standard deviation as calculated before (1.d))

    # TODO: Code to start on
    # histogram of duration

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
    ax.set_title('Distribution of Duration Monthly')
    plt.plot(base[:-1], values, c='blue')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.vlines(quantile_25, 0, 0.07, linestyles='dashed', label='25% Quantile', colors='green')
    plt.vlines(quantile_50, 0, 0.07, linestyles='dashed', label='50% Quantile', colors='yellow')
    plt.vlines(quantile_75, 0, 0.07, linestyles='dashed', label='75% Quantile', colors='red')
    plt.vlines(quantile_95, 0, 0.07, linestyles='dashed', label='95% Quantile')
    plt.legend(loc='upper right')
    io.save_fig(fig, p_filename="DurationMinutes_Distribution_Monthly.png", p_sub_folder2="math")


def corr_analysis(p_df):
    """Plot correlation between features.

    Args:
        p_df (DataFrame): DataFrame of trips in nuremberg
    Returns:
        no return
    """
    corrs = p_df.corr()
    # corrs.to_csv(io.get_path("feature_correlations.csv", "output"), sep=";")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corrs, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corrs, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(
        io.get_path(
            p_filename="Correlation.png",
            p_io_folder="output",
            p_sub_folder1="data_plots",
            p_sub_folder2="math"
        )
    )


def plot_mean_duration(p_df):
    """Plot the mean duration for each day of year and visualize the seasons.

    Args:
        p_df (DataFrame): Dataframe of trips in nuremberg
    Returns:
        no return
    """
    # calculate mean duration of trips for each day of year
    df_day_mean = p_df.groupby(by="dayofyear_start").mean()[["Duration", "Season"]]
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
    df_datapoints.drop("Duration_x", axis=1, inplace=True)
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


def plot_true_vs_predicted(p_y_true, p_y_predict, p_model_name):
    """Plot the true duration of trips against the predicted duration.

    Plot model predictions against the real duration values of trips.
    Args:
        p_y_true (Series): Series of true durations of trips
        p_y_predict (Series): Series of predicted durations of trips by model
        p_model_name (str): String of models name
    Returns:
        no return
    """
    # true vs predicted value
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
    ax_scatter.set_xlabel('True Y')
    ax_scatter.set_ylabel('Predicted Y')
    ax_scatter.set_title(p_model_name)
    ax_scatter.scatter(p_y_true, p_y_predict)
    io.save_fig(
        fig_scatter,
        p_filename=p_model_name+"_pred_vs_true.png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )

    # distribution of true vs distribution of predicted value
    p_y_predict = p_y_predict.flatten()  # NN gives 2-d array as predicted values

    fig_distr, ax_distr = plt.subplots(figsize=(10, 5))
    ax_distr.set_xlabel('Duration of Booking [min]')
    ax_distr.set_ylabel('Percentage [%]')
    ax_distr.set_title("Distribution of Predicted and True Durations")
    pred_values, pred_base = np.histogram(
        p_y_predict,
        bins=int(p_y_predict.max()),
        range=(0, int(p_y_predict.max())),
        weights=np.ones(len(p_y_predict)) / len(p_y_predict)
    )
    true_values, true_base = np.histogram(
        p_y_true,
        bins=int(p_y_predict.max()),
        range=(0, int(p_y_predict.max())),
        weights=np.ones(len(p_y_true)) / len(p_y_true)
    )
    ax_distr.plot(pred_base[:-1], pred_values, c='red', label=p_model_name)
    ax_distr.plot(true_base[:-1], true_values, c='green', label="True")
    plt.legend(loc='upper right')
    io.save_fig(
        fig_distr,
        p_filename=p_model_name+"_distribution.png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )


# TODO: add docstring
def plot_train_loss(p_history):
    """Plot the train and validation loss of Neural Network.

    Args:
        p_history (Object): History of loss during training of neural network
    Returns:
        no return
    """
    # Plotting the training and validation loss
    loss = p_history.history['loss']
    val_loss = p_history.history['val_loss']

    epochs = range(1, len(loss) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss, 'bo', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.set_title('Training and validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    io.save_fig(fig, "NN_error_per_epoch.png", p_sub_folder2="math")


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
