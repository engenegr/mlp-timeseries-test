# Copied from https://github.com/avanwyk/tensorflow-projects/blob/master/lr-finder/lr_finder.py
# Apache License 2.0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback



INPUT_SHAPE = 187


class LRFinder(Callback):
    """`Callback` that exponentially adjusts the learning rate after each training batch between `start_lr` and
    `end_lr` for a maximum number of batches: `max_step`. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the `plot` method.
    """

    def __init__(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        max_steps: int = 100,
        smoothing=0.9,
    ):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (
            step * 1.0 / self.max_steps
        )

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.0e"))
        ax.plot(self.lrs, self.losses)


def pretty_plot(history, field, fn, file=None):

    def plot(data, val_data, best_index, best_value, title, output=None):
        plt.plot(range(1, len(data) + 1), data, label="train")
        plt.plot(range(1, len(data) + 1), val_data, label="validation")
        if not best_index is None:
            plt.axvline(x=best_index + 1, linestyle=":", c="#777777")
        if not best_value is None:
            plt.axhline(y=best_value, linestyle=":", c="#777777")
        plt.xlabel("Epoch")
        plt.ylabel(field)
        plt.xticks(range(0, len(data), 20))
        plt.title(title)
        plt.legend()
        plt.show()

    data = history.history[field]
    val_data = history.history["val_" + field]
    tail = int(0.15 * len(data))

    best_index = fn(val_data)
    best_value = val_data[best_index]

    plot(
        data,
        val_data,
        best_index,
        best_value,
        "{} over epochs (best {:06.4f})".format(field, best_value)
    )
    plot(
        data[-tail:],
        val_data[-tail:],
        None,
        best_value,
        "{} over last {} epochs".format(field, tail)
    )


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def get_base_model(categories=2):
    return keras.Sequential(
        [
            keras.layers.Dense(
                15,
                activation="relu",
                input_shape=[INPUT_SHAPE],
                kernel_regularizer=regularizers.l2(0.0001),
            ),
            keras.layers.Dense(
                10, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
            ),
            keras.layers.Dense(5, activation="relu"),
            (
                keras.layers.Dense(1, activation="sigmoid")
                if categories == 2
                else keras.layers.Dense(5, activation="softmax")
            ),
        ]
    )
