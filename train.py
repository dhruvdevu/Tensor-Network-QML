import pyswarm_training
import noisyopt_training
import spsa
import test
import argparse
import utils
import numpy as np

# import model_full_simulate_wf as model
import model_qpu_cascade as model2
import model_qpu_balanced as model

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument("--savefile", help="filename to save params", type=str)
parser.add_argument("--resumefile", help="filename to resume params from", type=str)
# parser.add_argument("--data", help="bw, grey, or mnist", type = str, default = 'mnist')
parser.add_argument("--method", help="pyswarm or spsa", type = str, default = 'spsa')
parser.add_argument("--model", help="balanced or cascade", type=str, default='balanced')
args = parser.parse_args()
classes = (0,1)
train_data, train_labels, test_data, test_labels = utils.load_mnist_data('data/4', classes, val_split=0.1)
print(len(train_data), len(test_data))
#TODO: make hyperparameters also arguments to train()
params = None
n = 16#len(train_data[0])
n_trials = 1
mod = model.Model(n=n, num_trials=n_trials, classes=classes)
if args.model == 'cascade':
    mod = model2.Model(n=n, num_trials=n_trials, classes=classes)
# print(np.linalg.norm(train_data[0]), train_data[0], "hello")
init_params= None
init_params = np.random.normal(loc=0.0, scale=1.0, size=mod.count)

if args.resumefile:
    init_params = utils.load_params(args.resumefile)

test.get_accuracy(test_data[:200], test_labels[:200], mod, init_params)
if args.method == 'spsa':
    params, mod = spsa.train(train_data, train_labels, mod, params=init_params)
elif args.method == 'pyswarm':
    params, mod = pyswarm_training.train(train_data, train_labels, mod)
else:
    print("invalid optimization method")
    exit(0)


np.savetxt("params/" + args.savefile, params, delimiter = ",")
test.get_accuracy(test_data[:200], test_labels[:200], mod, params)
