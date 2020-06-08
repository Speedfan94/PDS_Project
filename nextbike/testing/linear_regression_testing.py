import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def test_regression_model(p_components, p_y_train):
    # TODO: Docstring
    df_metrics = pd.DataFrame(columns=["Components", "RMSE", "MAE", "R^2"])
    X_train_transformed = pd.DataFrame(p_components)
    for i in np.arange(0, len(X_train_transformed.columns)):
        print(".", end="")
        for j in np.arange(0, len(X_train_transformed.columns)):
            if i <= j:
                tmp = np.arange(start=i, stop=j+1)
            else:
                tmp = np.arange(start=j, stop=i+1)
            metric = train_linear_regression(X_train_transformed[tmp.astype(str)], p_y_train)
            df_metrics = df_metrics.append(metric, ignore_index=True)
    print()
    print("Min RMSE", df_metrics[df_metrics["RMSE"] == df_metrics["RMSE"].min()].iloc[0])
    print("Min MAE", df_metrics[df_metrics["MAE"] == df_metrics["MAE"].min()].iloc[0])
    print("Max R^2", df_metrics[df_metrics["R^2"] == df_metrics["R^2"].max()].iloc[0])


def train_linear_regression(p_components, p_y_train):
    """Train Linear Regression Model

    Train and save a Linear Regression model. Then evaluate the error metrics by another method.

    Args:
        p_X_train_scaled (DataFrame): Scaled X input of train set (matrix)
        p_y_train (Series): y output to train on (vector)
    Returns:
        No return
    """
    lin = LinearRegression()
    lin.fit(p_components, p_y_train)
    y_prediction = lin.predict(p_components)
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(p_y_train, y_prediction))
    # MAE
    mae = metrics.mean_absolute_error(p_y_train, y_prediction)
    # R^2
    r_squared = metrics.r2_score(p_y_train, y_prediction)

    return {"Components": list(p_components.columns), "RMSE": rmse, "MAE": mae, "R^2": r_squared}
