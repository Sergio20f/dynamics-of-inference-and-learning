import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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


def plot_fits(data_index_array, loss_array, params: list, power_or_exp: str, experiment_number,
              save="/home/sergiocalvo/Documents/qmul-internship/project/results_figs/"):
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
    :param experiment_number: Number of the experiment.
    :type experiment_number: int
    :param save: If not False, directory name to save the figures displaying the results of the experiments.

    :return: Corresponding plot.
    """
    
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_-%H_%M_%S")
    
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
    
    if save:
        plt.savefig(save + dt_string + f"_experiment_{experiment_number}")


def img_to_pdf(sdir:str, out_name:str, output_path=False):
    """
    Helper function to put together all of the images in a folder into a single pdf file.
    
    :param sdir: Directory where the target images are allocated.
    :type sdir: str
    :param out_name: Desired filename for the output pdf document.
    :type out_name: str
    :param output_path: Boolean parameter. If True, the function will print the output path.
    :type output_path: bool
    
    :return: None
    """
    
    pdf = FPDF()
    pdf.set_auto_page_break(0)
    
    img_list = [x for x in os.listdir(sdir) if x[0] != "."]
    
    for img in img_list:
        pdf.add_page()
        pdf.image(sdir+img)
    
    pdf.output(out_name+".pdf")
    
    if output_path:
        print(os.getcwd())