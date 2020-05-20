import pandas as pd
from .. import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    weekend = df["Weekend"].astype(int)
    df.drop(["p_spot_start", "p_bike_start", "p_spot_end", "p_bike_end", "Weekend"], axis=1, inplace=True)
    df_dummies = pd.concat([df, p_spot_start, p_bike_start, p_spot_end, p_bike_end, weekend], axis=1)
    return df_dummies


def cast_datetime(pDf):
    pDf["Start Time"] = pd.to_datetime(pDf["Start Time"], format="%Y-%m-%d %H:%M:%S").values.astype(int)
    pDf["End Time"] = pd.to_datetime(pDf["End Time"], format="%Y-%m-%d %H:%M:%S").values.astype(int)

    return pDf


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
    plt.savefig(io.get_path(filename="Correlation.png", io_folder="output", subfolder="data_plots"))


# TODO:remember to save the trained scaler for use on test data
def scale(pX_train):
    """Scale all independent variables in DataFrame

    Args:
        pX_train (DataFrame): DataFrame of independent variables

    Returns:
        X_train_scaled (DataFrame): DataFrame with scaled values
    """

    st_scaler = StandardScaler()

    # fit scaler on training set not on test set
    st_scaler.fit(pX_train)
    io.save_object(st_scaler, "Standard_Scaler.pkl")
    X_train_scaled = st_scaler.transform(pX_train)

    return X_train_scaled


def do_pca(pX_scaled_train):
    """Do a PCA to analyse which features to take for further predictions.

    Args:
        df (DataFrame): DataFrame of scaled data
    Returns:
        df (DataFrame): DataFrame with PCAs
    """
    # df = df[["Duration", "month", "day", "hour"]]
    pca = PCA(n_components=7)
    pca.fit(pX_scaled_train)
    print("Var explained:",pca.explained_variance_ratio_)
    print("Sum var explained", sum(pca.explained_variance_ratio_))

    io.save_object(pca, "PCA.pkl")
    X_train_scaled_pca = pca.transform(pX_scaled_train)

    return X_train_scaled_pca
