#From Wuggins QML_Trees
import os, pickle, random, gzip
import numpy as np
import skimage.transform
import tensorflow as tf

from tqdm import tqdm

TOTAL_NUM_CLASSES = 10 # MNIST specific
DIM = 4 #Size of image  DIM*DIM
DATA_DIR = 'data/'
class DataGenerator:
    def __init__(self):
        self.raw_train_data = self.raw_train_labels = self.raw_test_data = self.raw_test_labels = None

    def load_data(self, classes=None):
        """Function that supervises the data loading and processing."""
        (self.raw_train_data,
         self.raw_train_labels,
         self.raw_test_data,
         self.raw_test_labels) = get_raw_mnist()

        self.processed_images_train, self.labels_train = self.preprocess_data(data_type="train", classes=classes)
        self.processed_images_test, self.labels_test = self.preprocess_data(data_type="test", classes=classes)

        return self.processed_images_train, self.labels_train, self.processed_images_test, \
               self.labels_test

    def num_batches(self, data_type="train"):
        if data_type == "train":
            return len(self.processed_images_train) // CONFIG[data_type].num_images
        elif data_type == "test":
            return len(self.processed_images_test) // CONFIG[data_type].num_images
        else:
            raise ValueError("data_type must be \"train\" or \"test\"")

    def next_batch(self, data_type="train"):
        """
        Generator that allows for iterating over (randomized) batches of data.
        """
        if data_type == "train":
            processed_images, labels = self.processed_images_train, self.labels_train
        elif data_type == "test":
            processed_images, labels = self.processed_images_test, self.labels_test
        else:
            raise ValueError("data_type must be \"train\" or \"test\"")

        # Shuffle arrays in unison
        random_perm = np.random.permutation(processed_images.shape[0])
        processed_images, labels = processed_images[random_perm], labels[random_perm]

        for i in range(0, len(processed_images), CONFIG[data_type].num_images):
            batch_images = processed_images[i : i + CONFIG[data_type].num_images]
            batch_labels = labels[i : i + CONFIG[data_type].num_images]

            yield batch_images, batch_labels

    def preprocess_data(self, data_type="train", classes=None):
        """
        Preprocesses the data by downsampling the desired dataset (training or test) to the required
        size and then featurizing the data according to the given feature mapping.
        """
        # Default feature mapping is the built-in trigonometric mapping
        # if CONFIG["arch"].featurization == "trig":
        #     assert CONFIG["arch"].feature_size == 2
        #     featurize = self.trigonometric_featurize
        # elif CONFIG["arch"].featurization == "fourier":
        #     assert CONFIG["arch"].feature_size == 1
        #     featurize = self.random_fourier
        # else:
        #     raise NotImplementedError("Unknown featurization")

        # Filter dataset for images that have the desired labels.
        images_and_labels, classes = self.downsample(data_type=data_type, classes=classes)


        # Now process each image
        classes = sorted(list(classes))
        processed_images, labels = [], []
        for image, label in images_and_labels:
            # Iterate over the pixels, uniformly mapping onto an angle followed by featurizing.
            new_shape = (DIM, DIM)
            image = np.reshape(image, (28, 28))
            shrunk_image = skimage.transform.resize(image / 255.0, new_shape, anti_aliasing = True, mode = 'constant')
            shrunk_image = np.reshape(shrunk_image, (-1))

            # featurized_image = featurize(shrunk_image)
            # For now, no featurization
            processed_images.append(shrunk_image)

            # new_label = one_hot(
            #     index=classes.index(np.argmax(label)),
            #     length=len(classes))
            # Pick label instead of one-hot (TODO: remove 1-hot)
            new_label = list(label).index(1)

            labels.append(new_label)

        # Repackage into numpy arrays and return
        processed_images, labels = np.array(processed_images), np.array(labels)
        return processed_images, labels

    def downsample(self, data_type="train", classes=set([0, 1])):
        """
        A function for pruning the data and labels belonging to unwanted classes.
        """
        assert len(classes) == 2
        if data_type == "train":
            X, y = self.raw_train_data, self.raw_train_labels
        elif data_type =='test':
            X, y = self.raw_test_data, self.raw_test_labels
        else:
            raise NotImplementedError('Other data types not yet implemented.')

        # if classes is None: # Label dataset using the mode specified in config
        #     classes = set([int(c) for c in CONFIG["data"].classes_to_use.split(",")])
        #
        # if len(classes) == 1: # Do one vs. all classification
        #     c = list(classes)[0]
        #     images_and_labels = [(X[index], one_hot(1 if int(np.argmax(y[index])) == c else 0))
        #                          for index in range(len(X))]
        #     classes = set([0, 1])
        # else: # Prune images by those which don't have specified labels

        images_and_labels = [(X[index], y[index])
                             for index in range(len(X))
                             if int(np.argmax(y[index])) in classes]




        return images_and_labels, classes

    def trigonometric_featurize(self, image):
        """
        Built in trigonometric feature mapping, sending a normalized pixel to the vector
        [cos(value), sin(value)].
        """
        pixels = np.nditer(image)
        featurized_image = [(np.cos(pixel * np.pi / 2), np.sin(pixel * np.pi / 2))
                            for pixel in pixels]
        return featurized_image

    def random_fourier(self, image):
        """
        Featurizes the image vector by passing it through a random Fourier kernel, which maps a
        vector x to the vector
            cos(\Omega x + b)

        """
        if not hasattr(self, "_built_random_fourier") or not self._built_random_fourier:
            self._built_random_fourier = True

            dim = len(image)
            self._random_fourier_inp = tf.placeholder(tf.float32, shape=[None, dim])
            kernel = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
                input_dim=dim,
                output_dim=dim,
                stddev=5.0)
            self._random_fourier_out = kernel.map(self._random_fourier_inp)

        with tf.Session() as sess:
            image = np.array([image])
            out = sess.run(self._random_fourier_out, feed_dict={self._random_fourier_inp: image})

        return out[0]

def get_raw_mnist():
    """
    Loads raw MNIST data into memory, stored as pickled numpy arrays in a gz file.

    @return Four numpy arrays, the first holding training data, the second the
    training labels, the third the test data, and the fourth the test labels.
    """
    data_dir = DATA_DIR
    # test_set = CONFIG["data"].test_set
    assert os.path.exists(data_dir)
    path = os.path.join(data_dir, "mnist.pkl.gz")

    with gzip.open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train, val, test = u.load()

    def process_raw_data(unpickled):
        """
        A helper function to modify the format of the raw data before further processing.
        """
        X = unpickled[0] * 255
        y = unpickled[1]
        one_hot_y = []
        for index in range(len(y)):
            one_hot_y.append(one_hot(y[index], 10))

        y = np.asarray(one_hot_y)
        print(X.shape)
        print(y.shape)

        return X, y

    train_data, train_labels = process_raw_data(train)

    # if (test_set == "cross-validation"):
    #     train_data, train_labels, test_data, test_labels = split_data(
    #         train_data, train_labels)
    #
    # elif (test_set == "validation"):
    #     test_data, test_labels = process_raw_data(val)
    #
    # elif (test_set == "test"):
    #     test_data, test_labels = process_raw_data(test)
    #
    # else:
    #     raise ValueError("Invalid input for test_set option.")
    test_data, test_labels = process_raw_data(val)

    return train_data, train_labels, test_data, test_labels

def split_data(data, labels, split=0.85):
    """
    Given numpy arrays of data and corresponding labels, splits both, along axis 0, into a
    smaller (by the given factor) training data array and test data array, and corresponding
    arrays for the labels.

    @param data: Numpy array containing data.
    @param labels: Numpy array containing labels.
    @param split: Proportion of data to keep as training data.

    @return Four numpy arrays - train data, train labels, test data, test labels
    """
    assert 0 < split and split <= 1.0
    assert data.shape[0] == labels.shape[0]

    # Shuffle arrays in unison
    random_perm = np.random.permutation(data.shape[0])
    data, labels = data[random_perm], labels[random_perm]

    ind = int(split * data.shape[0])
    return data[ : ind], labels[ : ind], data[ind : ], labels[ind : ]

def one_hot(index, length=TOTAL_NUM_CLASSES):
    """
    A helper function to convert an integer to a one-hot vector.
    """
    ret = np.zeros(shape=(length,))
    ret[index] = 1.0
    return ret
