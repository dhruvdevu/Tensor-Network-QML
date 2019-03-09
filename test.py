import numpy as np
import math
# import model_full_simulate_wf as model
import model
import utils
import argparse
from tqdm import tqdm


def get_accuracy(test_data, test_labels, mod, params):
    num_correct = 0
    for i in tqdm(range(len(test_data))):
        dist =  mod.get_distribution(params, test_data[i].flatten())
        print(dist)
        label = 0
        if test_labels[i][0] == 0:
            label = 1
        if dist[label] > 0.5:
            num_correct += 1
    print("test accuracy = ", "{0:.2f}%".format(num_correct/len(test_data) * 100))
    print("number of circuit runs = ", mod.num_runs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test accuracy for provided params')
    parser.add_argument("--file", help="filename of params", type=str)
    parser.add_argument("--class1", help="class of data", type = int)
    parser.add_argument("--class2", help="class of data", type = int)

    args = parser.parse_args()
    filename = args.file
    print(filename)
    train_data, train_labels, test_data, test_labels = utils.load_mnist_data('data/4', (args.class1, args.class2))
    params = np.loadtxt(open("params/" + filename, "rb"), delimiter = ",")

    n = 16
    mod = model.Model(n=n, num_trials=1, classes=[args.class1, args.class2])
    get_accuracy(test_data[:50], test_labels[:50], mod, params)
