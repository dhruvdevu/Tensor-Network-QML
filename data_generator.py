import numpy as np
import math

BLACK_AND_WHITE = True
def generate_data():
    """Generate binary labelled data of the form [0, num, num, 0] and [num, num, 0, 0]"""
    data = []
    labels = []

    for i in range (0, 100):
        if BLACK_AND_WHITE:
            x = 0.5
        else:
            x = np.random.random_sample()
        data.append([x, 1-x, 0, 0])
        labels.append(0)

    for j in range(100, 200):
        if BLACK_AND_WHITE:
            x = 0.5
        else:
            x = np.random.random_sample()
        data.append([0, x, 1-x, 0])
        labels.append(1)
    n = len(data[0])
    labels = np.array(labels)
    combined = np.hstack((data, labels[np.newaxis].T))
    np.random.shuffle(combined)
    data = combined[:,list(range(n))]
    labels = combined[:,n]
    num_training = math.floor(0.7*len(data))
    train_data = data[:num_training]
    train_labels = labels[:num_training]
    test_data = data[num_training:]
    test_labels = labels[num_training:]

    if BLACK_AND_WHITE:
        np.savetxt("data/train_data_bw.csv", train_data, delimiter = ",")
        np.savetxt("data/train_labels_bw.csv", train_labels, delimiter = ",")
        np.savetxt("data/test_data_bw.csv", test_data, delimiter = ",")
        np.savetxt("data/test_labels_bw.csv", test_labels, delimiter = ",")
    else:
        np.savetxt("data/train_data_grey.csv", train_data, delimiter = ",")
        np.savetxt("data/train_labels_grey.csv", train_labels, delimiter = ",")
        np.savetxt("data/test_data_grey.csv", test_data, delimiter = ",")
        np.savetxt("data/test_labels_grey.csv", test_labels, delimiter = ",")

def read_data():
    if BLACK_AND_WHITE:
        print(np.loadtxt(open("data/train_data_bw.csv", "rb"), delimiter = ","))
        print(np.loadtxt(open("data/train_labels_bw.csv", "rb"), delimiter = ","))
    else:
        print(np.loadtxt(open("data/train_data.csv", "rb"), delimiter = ","))
        print(np.loadtxt(open("data/train_labels.csv", "rb"), delimiter = ","))


if __name__ == "__main__":
    generate_data()
    # read_data()
