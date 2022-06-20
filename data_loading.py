import tensorflow as tf
import tensorflow_datasets as tfds
from helpers import normalize_img


class Data:
    """
    Class to import a specific dataset and build training, test, and validation datasets.

    Attributes:
    -----------
    name (str) : tensorflow_datasets dataset name.
    batch_size (int) = integer value for the batch size, default is 128.

    Methods:
    -----------
    test_data_prep(): generates independent and different training and validation datasets of normalised images and
    applies cache(), batch(), and prefetch().
    train_data_prep(): generates a training dataset of normalised images and applies cache(), batch(), and prefetch().
    """

    def __init__(self, name:str, batch_size=128, norm_func=normalize_img, resize=False):
        """
        Initialises the class and creates the global variables.

        :param name: tensorflow_datasets dataset name.
        :type name: str
        :param batch_size: integer value for the batch size, default is 128.
        :type batch_size: int
        :param resize: Parameter corresponding to the new dimensions of the images contained in the dataset after resizing. If 
                       False then the images in the dataset are not resized.
        """
        self.batch_size = batch_size
        self.name = name
        self.norm_func = norm_func
        self.resize = resize

        pass

    def test_data_prep(self):
        """
        Method that generates independent and different training and validation datasets of normalised images and
        applies cache(), batch(), and prefetch().

        :return: A tuple containing the validation data and test data respectively.
        """
        validation_data = tfds.load(self.name,
                                    split="validation[:50%]",
                                    shuffle_files=True,
                                    as_supervised=True,
                                    with_info=False)

        test_data = tfds.load(self.name, split="validation[-50%:]",
                              shuffle_files=True,
                              as_supervised=True,
                              with_info=False)

        if self.norm_func:
            # Map the normalisation function
            validation_data = validation_data.map(self.norm_func,
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_data = test_data.map(self.norm_func,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.resize:
            validation_data = validation_data.map(lambda image, label: (tf.image.resize(image, [self.resize[0], 
                                                                          self.resize[1]]),
                                                                          label))
            
            test_data = test_data.map(lambda image, label: (tf.image.resize(image, [self.resize[0], self.resize[1]]),
                                                              label))

        # Batch the sets
        validation_data = validation_data.batch(self.batch_size)
        test_data = test_data.batch(self.batch_size)

        # Cache and Prefetch sets
        validation_data = validation_data.cache()
        validation_data = validation_data.prefetch(tf.data.experimental.AUTOTUNE)
        test_data = test_data.cache()
        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

        return validation_data, test_data

    def train_data_prep(self, dts: int, data_type="image"):
        """
        Method that generates a training dataset of normalised images and applies cache(), batch(), and prefetch(). It
        uses the method "ReadInstruction()" from tfds.core to load the desired amount of data in each training
        iteration. This desired amount of data is determined by the parameter dts.

        :param dts: Parameter that determines the quantity of data going into the output training dataset.
        :type dts: int
        :param data_type: (Will use in the future to determine the datatype of the data points in the target dataset).
        :type data_type: str
        :param norm_func: Takes the value of None for whenever we do not want to apply any normalisation function to our
        data. If not None, then it must be a python function.
        :type norm_func: function

        :return: Training data.
        """
        train_data, ds_info = tfds.load(self.name,
                                        split=tfds.core.ReadInstruction("train",
                                                                        from_=0, to=dts, unit="abs"),
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True)

        if self.norm_func:
            train_data = train_data.map(self.norm_func,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.resize:
            train_data = train_data.map(lambda image, label: (tf.image.resize(image, [self.resize[0], self.resize[1]]), label))
        
        # Prepare training data for fitting
        train_data = train_data.shuffle(ds_info.splits["train"].num_examples)

        if self.batch_size <= 0:
            print("Something is wrong with the batch size")
            return None

        else:
            train_data = train_data.cache()
            train_data = train_data.batch(self.batch_size)
            train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        return train_data
