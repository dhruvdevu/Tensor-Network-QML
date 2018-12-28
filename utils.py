import numpy as np

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
