import numpy as np
import math
import model
import utils
import argparse
from tqdm import tqdm


def get_accuracy(test_data, test_labels, mod, params):
    num_correct = 0
    for i in tqdm(range(len(test_data))):
        dist =  mod.get_distribution(params, test_data[i])
        if dist[math.floor(test_labels[i])] > 0.5:
            num_correct += 1
    print("test accuracy = ", "{0:.2f}%".format(num_correct/len(test_data) * 100))
    print("number of circuit runs = ", mod.num_runs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test accuracy for provided params')
    parser.add_argument("--file", help="filename of params", type=str)
    parser.add_argument("--data", help="bw, grey, or mnist", type = str, default = 'mnist')

    args = parser.parse_args()
    filename = args.file
    print(filename)
    train_data, train_labels, test_data, test_labels = utils.load_data(args.data)
    params = np.loadtxt(open("params/" + filename, "rb"), delimiter = ",")

    n = len(test_data[0])
    mod = model.Model(n=n, num_trials=1)
    get_accuracy(test_data[:50], test_labels[:50], mod, params)
