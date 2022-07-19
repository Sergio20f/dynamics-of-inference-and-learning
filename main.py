import os
import sys

sys.path.append("../")

import tensorflow as tf

tf.random.set_seed(42)

# fix bug for the correct use of Conv2D layers
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from config_loader import Config
from helpers import normalize_img, mask_to_categorical
from model_setup import training_fit_loop

# Accessing config.yaml
config = Config("config.yaml")

# Defining the model
base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)

inputs = tf.keras.layers.Input(shape=(32, 32, 3), name="input_layer")
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax",
                                name="output_layer")(x)

model_0 = tf.keras.Model(inputs, outputs)

# Training loop
a = training_fit_loop(model=model_0,
                      model_params=(config.INPUT_SHAPE, config.OPTIMIZER, config.LOSS, [config.METRICS]),
                      data_step=config.DATA_STEP,
                      n=config.N,
                      data_loading_params=(config.NAME, config.BATCH_SIZE, config.NORM_FUNC, config.RESIZE,
                                           config.CUSTOM_DIR, config.VALIDATION_OR_TEST),
                      start_data=config.START_DATA,
                      save_df=os.getcwd()[:-len("tests")] + "results_df/",
                      verbose=config.VERBOSE)
