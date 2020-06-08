import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from nextbike import io


def test_neuralnetwork_model(p_components, p_y_train):
    # TODO: Docstring
    # p_components=p_components.iloc["1", "2"]
    df_metrics = pd.DataFrame(columns=["Components", "RMSE", "MAE", "R^2"])
    X_train_transformed = pd.DataFrame(p_components)
    for i in np.arange(0, len(X_train_transformed.columns)):
        print(".", end="")
        for j in np.arange(0, len(X_train_transformed.columns)):
            print()
            print(i, " ", j)
            print()
            if i <= j:
                tmp = np.arange(start=i, stop=j+1)
            else:
                tmp = np.arange(start=j, stop=i+1)
            metric = train_neural_network(X_train_transformed[tmp.astype(str)], p_y_train)
            df_metrics = df_metrics.append(metric, ignore_index=True)
    print()
    io.save_csv(df_metrics,"NN_Metrics.csv")
    print("Min RMSE", df_metrics[df_metrics["RMSE"] == df_metrics["RMSE"].min()].iloc[0])
    print("Min MAE", df_metrics[df_metrics["MAE"] == df_metrics["MAE"].min()].iloc[0])
    print("Max R^2", df_metrics[df_metrics["R^2"] == df_metrics["R^2"].max()].iloc[0])


def train_neural_network(p_components, p_y_train):
    """Train Neural_network Model

    Train and save a Linear Regression model. Then evaluate the error metrics by another method.

    Args:
        p_components (DataFrame): Scaled X input of train set (matrix)
        p_y_train (Series): y output to train on (vector)
    Returns:
        No return
    """
    neural_network = keras.Sequential(
            [layers.Dense(36, activation="relu", input_shape=[p_components.shape[1]]),
             # layers.Dropout(0.2),
             layers.Dense(36, activation="relu"),
             # layers.Dense(36, activation="softmax"),
             # layers.Dense(36, activation="softmax"),
             # layers.Dropout(0.2),
             layers.Dense(1)])
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    neural_network.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae", "mse"]
    )
    epochs = 10
    # batch_size = 200  # right now not used but should be tried
    history = neural_network.fit(p_components, p_y_train.values, epochs=epochs, validation_split=0.2)
    y_prediction = neural_network.predict(p_components)

    # -----------------------------------------------------------------------

    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(p_y_train, y_prediction))
    # MAE
    mae = metrics.mean_absolute_error(p_y_train, y_prediction)
    # R^2
    r_squared = metrics.r2_score(p_y_train, y_prediction)

    return {"Components": list(p_components.columns), "RMSE": rmse, "MAE": mae, "R^2": r_squared}
