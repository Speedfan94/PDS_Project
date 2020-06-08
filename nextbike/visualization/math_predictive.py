from nextbike import io
import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_predicted(p_y_true, p_y_predict, p_model_name):
    """Plot the true duration of trips against the predicted duration.

    Plot model predictions against the real duration values of trips.

    Args:
        p_y_true (Series): Series of true durations of trips
        p_y_predict (Series): Series of predicted durations of trips by model
        p_model_name (str): String of models name
    Returns:
        No return
    """
    # true vs predicted value
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
    ax_scatter.set_xlabel("True Y")
    ax_scatter.set_ylabel("Predicted Y")
    ax_scatter.set_title(p_model_name)
    ax_scatter.scatter(p_y_true, p_y_predict)
    io.save_fig(
        fig_scatter,
        p_filename=p_model_name+"_pred_vs_true.png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig_scatter)

    # distribution of true vs distribution of predicted value
    p_y_predict = p_y_predict.flatten()  # NN gives 2-d array as predicted values

    fig_distr, ax_distr = plt.subplots(figsize=(10, 5))
    ax_distr.set_xlabel("Duration of Booking [min]")
    ax_distr.set_ylabel("Percentage [%]")
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
    ax_distr.plot(pred_base[:-1], pred_values, c="red", label=p_model_name)
    ax_distr.plot(true_base[:-1], true_values, c="green", label="True")
    plt.legend(loc="upper right")
    io.save_fig(
        fig_distr,
        p_filename=p_model_name+"_distribution.png",
        p_io_folder="output",
        p_sub_folder1="data_plots",
        p_sub_folder2="math"
    )
    plt.close(fig_distr)


def plot_train_loss(p_history):
    """Plot the train and validation loss of Neural Network.

    Args:
        p_history (Object): History of loss during training of neural network
    Returns:
        No return
    """
    # Plotting the training and validation loss
    loss = p_history.history["loss"]
    val_loss = p_history.history["val_loss"]

    epochs = range(1, len(loss) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss, "bo", label="Training loss")
    ax.plot(epochs, val_loss, "b", label="Validation loss")
    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.legend()
    io.save_fig(fig, "NN_error_per_epoch.png", p_sub_folder2="math")
    plt.close(fig)


# TODO: add docstring
def plot_direction_classification(p_X_train, p_y_train):
    p_X_train["Direction"] = p_y_train
    # x1 = p_X_train[p_X_train["Direction"]==True]["Duration"]
    y1 = p_X_train[p_X_train["Direction"] == True]["Dist_start"]
    # x2 = p_X_train[p_X_train["Direction"]==False]["Duration"]
    y2 = p_X_train[p_X_train["Direction"] == False]["Dist_start"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(0, y2, c="green")
    ax.scatter(0, y1, c="red")
    io.save_fig(fig, p_filename="Classification_Data.png", p_sub_folder2="math")
    plt.close(fig)
