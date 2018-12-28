import numpy as np


def load_data(BLACK_AND_WHITE=True):
    """This method loads the data and separates it into testing and training data."""
    if BLACK_AND_WHITE:
        train_data = np.loadtxt(open("data/train_data_bw.csv", "rb"), delimiter = ",")
        train_labels = np.loadtxt(open("data/train_labels_bw.csv", "rb"), delimiter = ",")
        test_data = np.loadtxt(open("data/test_data_bw.csv", "rb"), delimiter = ",")
        test_labels = np.loadtxt(open("data/test_labels_bw.csv", "rb"), delimiter = ",")

    else:
        train_data = np.loadtxt(open("data/train_data_grey.csv", "rb"), delimiter = ",")
        train_labels = np.loadtxt(open("data/train_labels_grey.csv", "rb"), delimiter = ",")
        test_data = np.loadtxt(open("data/test_data_grey.csv", "rb"), delimiter = ",")
        test_labels = np.loadtxt(open("data/test_labels_grey.csv", "rb"), delimiter = ",")
    #prep_state_program([7.476498897658023779e-01,2.523501102341976221e-01,0.000000000000000000e+00,0.000000000000000000e+00])
    #Number of qubits

    return (train_data, train_labels, test_data, test_labels)
