import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import scipy
from scipy import optimize
import pandas as pd
from datetime import datetime


from helpers import normalize_img, monoExp, powerlaw, plot_fits
from data_loading import Data


def build_and_compile(model=None, input_shape=(28, 28, 1), optimizer=Adam, lr=0.001,
                      loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]):
    """
    Function that builds the model instance and compiles it.
    In future work we will implement further customisation for the model architecture.

    :param model: Tensorflow model. If "None" then the function will construct a default simple CNN with input_shape.
    :param input_shape: Input shape as a tuple for the first convolutional layer of the model. Default: (28, 28, 1).
    :type input_shape: tuple
    :param optimizer: Optimizer to compile the model.
    :param lr: Learning rate.
    :type lr: float
    :param loss: Loss used to compile the model.
    :param metrics: Metrics to compute the performance of the model throughout training.

    :return: Compiled model for further use in training.
    """

    if model == None: # Default model
        # Build the model
        model = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=2, padding="same",
                    activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=2),
            Dropout(0.3),

            Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"),
            MaxPooling2D(pool_size=2),
            Dropout(0.3),

            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(10)
        ])

    # Compile the model
    model.compile(optimizer=optimizer(lr),
                  loss=loss,
                  metrics=metrics)

    return model


def step_train(model, train_data, test_data, validation_data, dts):
    """
    Simple function that fits a model (with the .fit() method) and evaluates it using input training, testing, and
    validation datasets. To be used n-times as a step within the training loop.

    :param model: Tensorflow defined and compiled model.
    :param train_data: Training data preferably in tf.data.Dataset() format (and cache(), batch(), and prefetch()
    applied).
    :param test_data: Testing data preferably in tf.data.Dataset() format (and cache(), batch(), and prefetch()
    applied).
    :param validation_data: Validation data preferably in tf.data.Dataset() format (and cache(), batch(), and prefetch()
    applied).
    :param dts: Datasize of the specific training iteration. This parameter will determine the relation between the size
    of the datasets and the epochs used during training.

    :return: Validation loss of the model after training.
    """

    epochs = int(240000 / dts) # 240000 tested constant appropriate for defining epochs

    # Fit the model
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=validation_data, verbose=0)

    loss_val = model.evaluate(test_data)[0]

    return loss_val


def training_fit_loop(model, train_data_name: str, data_step: int, n: int, data_loading_params: tuple, N=1, start_data=500,
                      plot=True, save_df=False):
    """
    Function that will put together the previously defined functions with the aim of generating a training loops with n
    iterations. It will also use the results of the training experiments and fit it to both an exponential and a power
    law function determining which one is a better fit. Lastly, it will generate a dataframe with the results of each
    experiment with some interesting metrics to look at.

    :param model: Tensorflow model to train. If None, then the function uses a default simple CNN.
    :param train_data_name: Name that the training data takes within tensorflow_datasets.
    :type train_data_name: str
    :param data_step: How many data points are gonna be added to the dataset after each training step.
    :type data_step: int
    :param n: Number of iterations for training in each experiment.
    :param n: int
    :param data_loading_params: Data loading parameters in specific order -> 1. name, 2. batch_sie, 3. resize, 4. custom_dir,
    5. val_or test
    :type data_loading_params: tuple
    :param N: Number of experiments.
    :type N: int
    :param start_data: Determines the size of the initial training dataset from 0 (Default: 500).
    :type start_data: int
    :param plot: Boolean parameter. If true then the function will plot fitting results. If false, it will just return
    the results.
    :type plot: bool
    :param save_df: If not False, path to folder where the dataframe will be saved as a pickle.
    :type save_df: False or str

    :return: R² of the fit, list of the fitting parameters, dataframe with useful information from the fitting process.
    """
    # Decompose data loading parameters
    name, batch_size, resize, custom_dir, validation_or_test = data_loading_params

    decay0 = []
    decay1 = []
    r2_0 = []
    r2_1 = []
    glob_params = []
    df = pd.DataFrame(columns=["R² Exp", "Exp function fit",
                               "Exp: Mean power constant", "Exp: Root of variance",
                               "Exp: R² value of fit",
                               "R² PL", "PL function fit",
                               "PL: Mean power constant", "PL: Root of variance",
                               "PL: R² value of fit"])

    data = Data(name=name, batch_size=batch_size, resize=resize, custom_dir=custom_dir)

    if custom_dir:
        test_data = data.test_data_prep(validation_or_test=validation_or_test)
        validation_data = test_data
    else:
        validation_data, test_data = data.test_data_prep(validation_or_test=validation_or_test)

    # Loop repeats the process for N experiments.
    for j in range(N):

        model = build_and_compile(model=model)

        loss_list = []
        data_index = []

        for i in range(n):
            data_size = int(start_data + i * data_step) # Adapts the size of the training dataset for each iteration
            train_data = data.train_data_prep(dts=data_size)

            loss_val = step_train(model, train_data, test_data, validation_data, dts=data_size)

            data_index.append(data_size)
            loss_list.append(loss_val)

        # Transform the lists into arrays
        loss_array = np.array(loss_list)
        data_index_array = np.array(data_index)

        # Fit for exponential
        p_exp = (0.1, 0.00001, 0.1)
        params_exp, cov_exp = scipy.optimize.curve_fit(monoExp, data_index_array, loss_array,
                                                       p_exp, maxfev=7000)
        m_exp, t_exp, b_exp = params_exp

        # Fit for power law
        p_power = (0.1, 0.00001, 0.1)
        param_power, cov_power = scipy.optimize.curve_fit(powerlaw, data_index_array,
                                                          loss_array, p_power, maxfev=7000)
        m_power, t_power, b_power = param_power

        # Determine the quality of the exponential fit and write out
        squared_diffs_exp = np.square(loss_array - monoExp(data_index_array, m_exp, t_exp, b_exp))
        squared_diffs_from_mean_exp = np.square(loss_array - np.mean(loss_array))
        r_sq_exp = 1 - np.sum(squared_diffs_exp) / np.sum(squared_diffs_from_mean_exp)
        decay0.append(t_exp)
        r2_0.append(r_sq_exp)

        # Determine quality of the powerlaw fit and write out
        squared_diffs_power = np.square(loss_array - powerlaw(data_index_array, m_power,
                                                              t_power, b_power))
        squared_diffs_from_mean_power = np.square(loss_array - np.mean(loss_array))
        r_sq_power = 1 - np.sum(squared_diffs_power) / np.sum(squared_diffs_from_mean_power)
        decay1.append(t_power)
        r2_1.append(r_sq_power)

        # Adding Experiment data to the main dataframe
        decay0_array = np.array(decay0)
        r2_0_array = np.array(r2_0)
        decay1_array = np.array(decay1)
        r2_1_array = np.array(r2_1)

        df.loc[f"Experiment {j+1}:"] = [r_sq_exp, (m_exp, t_exp, b_exp),
                                      np.mean(decay0_array), np.sqrt(np.var(decay0_array)),
                                      np.mean(r2_0_array),
                                      r_sq_power, (m_power, t_power, b_power),
                                      np.mean(decay1_array), np.sqrt(np.var(decay1_array)),
                                      np.mean(r2_1_array)]
        
        if N > 1 and plot:

            if r_sq_exp < r_sq_power:
                plot_fits(data_index_array, loss_array, [m_exp, t_exp, b_exp], "exp", experiment_number=j+1)
                glob_params.append([m_exp, t_exp, b_exp])

            else:
                plot_fits(data_index_array, loss_array, [m_power, t_power, b_power], "power", experiment_number=j+1)
                glob_params.append([m_power, t_power, b_power])
    
    if save_df:         
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_-%H_%M_%S")
        
        # Save the final dataframe as a pickle
        df.to_pickle(save_df + dt_string + f"_experiment_{j+1}")
    
    if N > 1:
        plot = False

    if r_sq_exp < r_sq_power and plot:
        plot_fits(data_index_array, loss_array, [m_exp, t_exp, b_exp], "exp", experiment_number=j+1)
        return r2_0, [m_exp, t_exp, b_exp], df

    elif r_sq_exp > r_sq_power and plot:
        plot_fits(data_index_array, loss_array, [m_power, t_power, b_power], "power", experiment_number=j+1)
        return r2_1, [m_power, t_power, b_power], df

    elif r_sq_exp < r_sq_power:
        return r2_0, glob_params, df

    else:
        return r2_1, glob_params, df
