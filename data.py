import pickle as pk
import numpy as np
import tensorflow as tf
import skimage.transform

class DataGenerator:
    def __init__(self):
        mnist_data = tf.keras.datasets.mnist.load_data()
        self.train_images = mnist_data[0][0]
        self.train_labels = mnist_data[0][1]
        self.test_images = mnist_data[1][0]
        self.test_labels = mnist_data[1][1]

    def shrink_images(self, new_shape):
        self.train_images = resize_images(self.train_images, new_shape)
        self.test_images = resize_images(self.test_images, new_shape)

    def featurize(self):
        self.train_images = trig_featurize(self.train_images)
        self.test_images = trig_featurize(self.test_images)

    def export(self, path):
        train_dest = path + '_train'
        test_dest = path + '_test'
        save_data(self.train_images, self.train_labels, train_dest)
        save_data(self.test_images, self.test_labels, test_dest)

def select_digits(images, labels, digits):
    cumulative_test = labels == digits[0]
    for digit in digits[1:]:
        digit_test = labels == digit
        cumulative_test = np.logical_or(digit_test, cumulative_test)
    valid_images = images[cumulative_test]
    valid_labels = labels[cumulative_test]
    return (valid_images, valid_labels)

def resize_images(images, shape):
    num_images = images.shape[0]
    new_images_shape = (num_images, shape[0], shape[1])
    new_images = skimage.transform.resize(
        images,
        new_images_shape,
        anti_aliasing = True,
        mode = 'constant')
    return new_images

def batch_generator(images, labels, batch_size):
    num_images = images.shape[0]
    random_perm = np.random.permutation(num_images)
    randomized_images = images[random_perm]
    randomized_labels = labels[random_perm]
    for i in range(0, num_images, batch_size):
        batch_images = randomized_images[i : i + batch_size]
        batch_labels = randomized_labels[i : i + batch_size]
        yield batch_images, batch_labels

def flatten_images(images):
    num_images = images.shape[0]
    flattened_image = np.reshape(images, [num_images, -1])
    return flattened_image

def trig_featurize(images):
    flat_images = flatten_images(images)
    (num_images, num_pixels) = flat_images.shape
    prep_axes = np.reshape(flat_images, (num_images, num_pixels, 1))
    pix_copy = np.tile(prep_axes, [1, 1, 2])
    pix_copy[:, :, 0] = np.cos(pix_copy[:, :, 0] * np.pi/2)
    pix_copy[:, :, 1] = np.sin(pix_copy[:, :, 1] * np.pi/2)
    return pix_copy

def split_data(images, labels, split):
    num_images = images.shape[0]
    random_perm = np.random.permutation(num_images)
    randomized_images = images[random_perm]
    randomized_labels = labels[random_perm]
    split_point = int(split * num_images)
    left_split = (images[:split_point], labels[:split_point])
    right_split = (images[split_point:], labels[split_point:])
    return (left_split, right_split)

def binary_labels(labels):
    max_digit = np.amax(labels)
    binary_values = np.floor_divide(labels, max_digit)
    binary_labels = one_hot(binary_values)
    return binary_labels

def one_hot(labels):
    length = np.amax(labels) + 1
    blank = np.zeros(labels.size * length)
    multiples = np.arange(labels.size)
    index_shift = length * multiples
    new_indices = labels + index_shift
    blank[new_indices] = 1
    matrix_blank = np.reshape(blank, [labels.size, length])
    return matrix_blank

def get_data(data_path, digits, val_split = 0):
    train_raw = load_data(data_path + '_train')
    test_raw = load_data(data_path + '_test')
    (train_images, train_labels_int) = select_digits(*train_raw, digits)
    (test_images, test_labels_int) = select_digits(*test_raw, digits)
    train_labels = binary_labels(train_labels_int)
    test_labels = binary_labels(test_labels_int)
    if val_split:
        (true_train_data, val_data) = split_data(train_images, train_labels, val_split)
    return (true_train_data, val_data, (test_images, test_labels))

def save_data(images, labels, path):
    dest = open(path, 'wb')
    data = (images, labels)
    pk.dump(data, dest)

def load_data(path):
    dest = open(path, 'rb')
    data = pk.load(dest)
    return data
