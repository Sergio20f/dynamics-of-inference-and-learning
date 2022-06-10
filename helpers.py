import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize_img(image, label):
    """
    Function to map across all the images in a dataset in order to normalise them (e.g. normalise pixel values and
    set the datatype to float32).

    :param image: Image to normalise.
    :type image: array
    :param label: Target label of the image.
    :type label: int

    :return: Normalised image array and its label as a tuple.
    """
    return tf.cast(image, tf.float32) / 255., label


def monoExp(x, m, t, b):
    """
    Mapping general exponential function.

    :param x: Coefficient x.
    :type x: float
    :param m: Coefficient m.
    :type m: float
    :param t: Coefficient t.
    :type t: float
    :param b: Coefficient b.
    :type b: float

    :return: A mapped value into an exponential space determined by the input coefficients.
    """
    return m * np.exp(-t * x) + b


def powerlaw(x, m, t, b):
    """
    Mapping general power law function.

    :param x: Coefficient x.
    :type x: float
    :param m: Coefficient m.
    :type m: float
    :param t: Coefficient t.
    :type t: float
    :param b: Coefficient b.
    :type b: float

    :return: A mapped value into a power law space determined by the input coefficients.
    """
    return m * x ** (-t) + b


def plot_random_sample(dataset):
    """
    Function that plots a random image from the input dataset.

    :param dataset: Tensorflow dataset containing images from which the random image will be sampled.
    :type dataset: tf.data.Dataset

    :return: Corresponding plot.
    """
    random_index = np.random.randint(0, len([i for i in dataset.take(1)][0][0]))

    if type(dataset) == tuple:
        plt.imshow([i for i in dataset[0].take(1)][0][random_index].numpy())

    else:
        plt.imshow([i for i in dataset.take(1)][0][0][random_index].numpy())


def plot_fits(data_index_array, loss_array, params: list, power_or_exp: str):
    """
    Plotting function that will plot the fitted curve (either exponential or power law) using a set of coefficients.

    :param data_index_array: Data on the horizontal axis of the plot. This should correspond to an array containing the
    number of data samples at each training iteration.
    :type data_index_array: array
    :param loss_array: Array containing the loss at the end of each training iteration (vertical axis for our fitted
    line).
    :type loss_array: array
    :param params: List of parameters to fit the data.
    :type params: list
    :param power_or_exp: Parameter that determines wheter the data will be fitted with an exponential ("exp") or a power
    law curve.
    :type power_or_exp: str

    :return: Corresponding plot.
    """
    # plot the results
    plt.figure()
    plt.plot(data_index_array, loss_array, '.', label="data")

    if power_or_exp == "exp":
        plt.plot(data_index_array, monoExp(data_index_array, params[0], params[1], params[2]), '--',
                 label="fitted exponential", color='green')

    else:
        plt.plot(data_index_array, powerlaw(data_index_array, params[0], params[1], params[2]), '--',
                 label="fitted powerlaw", color='red')

    plt.legend()
    plt.title("Fitted Curve")
