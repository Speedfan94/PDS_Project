import pandas as pd
from .. import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_dummies(df):
    """Create dummie values for all usefull non numerical features.

    Args:
        df (DataFrame): DataFrame of trips in nuremberg
    Returns:
        df_dummies (DataFrame): DataFrame with dummies instead of booleans
    """
    print("Start dummie creation...")
    p_spot_start = df["p_spot_start"].astype(int)
    p_bike_start = df["p_bike_start"].astype(int)
    p_spot_end = df["p_spot_end"].astype(int)
    p_bike_end = df["p_bike_end"].astype(int)
    df.drop(["p_spot_start", "p_bike_start", "p_spot_end", "p_bike_end"], axis=1, inplace=True)
    df_dummies = pd.concat([df, p_spot_start, p_bike_start, p_spot_end, p_bike_end], axis=1)
    return df_dummies


def corr_analysis(df):
    """Plot correlation between features

    Args:
        df (DataFrame): DataFrame of trips in nuremberg
    Returns:
        no return
    """
    corrs = df.corr()
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
    plt.savefig(io.get_path("Correlation.png", "output"))


# TODO:remember to save the trained scaler for use on test data
def scale(df):
    """Scale all features in DataFrame

    Args:
        df (DataFrame): DataFrame of trips in nuremberg
    Returns:
        df (DataFrame): DataFrame with scaled values
    """
    print()


def do_pca(df):
    """Do a PCA to analyse which features to take for further predictions.

    Args:
        df (DataFrame): DataFrame of scaled data
    Returns:
        df (DataFrame): DataFrame with PCAs
    """
    print("Start PCA...")
