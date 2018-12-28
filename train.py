import pyswarm_training
import noisyopt_training
import test
import argparse
import utils
import numpy as np

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument("--savefile", help="filename to save params", type=str)
parser.add_argument("--data", help="bw, grey, or mnist", type = str, default = 'mnist')
parser.add_argument("--method", help="pyswarm or spsa", type = str, default = 'spsa')
args = parser.parse_args()

train_data, train_labels, test_data, test_labels = utils.load_data(args.data)
#TODO: make hyperparameters also arguments to train()
params = None
mod = None
print(np.linalg.norm(train_data[0]), train_data[0], "hello")
if args.method == 'spsa':
    params, mod = noisyopt_training.train(train_data, train_labels)
elif args.method == 'pyswarm':
    params, mod = pyswarm_training.train(train_data, train_labels)
else:
    print("invalid optimization method")
    exit(0)

np.savetxt("params/" + args.savefile, params, delimiter = ",")
test.get_accuracy(test_data, test_labels, mod, params)
