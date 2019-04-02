import numpy as np
import data

DATA_DIR = 'data/'
def load_data(type):
    """This method loads the data and separates it into testing and training data."""
    if type in ['bw', 'grey', 'mnist']:
        train_data_str = DATA_DIR + 'train_data_' + type + '.csv'
        train_label_str = DATA_DIR + 'train_labels_' + type + '.csv'
        test_data_str = DATA_DIR + 'test_data_' + type + '.csv'
        test_label_str = DATA_DIR + 'test_labels_' + type + '.csv'
        train_data = np.loadtxt(open(train_data_str, "rb"), delimiter = ",")
        train_labels = np.loadtxt(open(train_label_str, "rb"), delimiter = ",")
        test_data = np.loadtxt(open(test_data_str, "rb"), delimiter = ",")
        test_labels = np.loadtxt(open(test_label_str, "rb"), delimiter = ",")

        return (train_data, train_labels, test_data, test_labels)
    else:
        print("Invalid data choice")

def save_params(params):
    np.savetxt("params/temp.csv", params, delimiter = ",")

def load_params(filename):
    return np.loadtxt(open("params/" + filename, "rb"), delimiter = ",")

def load_mnist_data(data_path, digits, val_split=0.7):
    (train_samples, val_samples, test_samples) = data.get_data(data_path, digits, val_split)
    (train_data, train_labels) = train_samples
    (val_data, val_labels) = val_samples
    (test_data, test_labels) = test_samples
    return (train_data, train_labels, test_data, test_labels)
