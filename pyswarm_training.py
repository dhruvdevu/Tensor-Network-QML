import numpy as np
import math
from pyswarm import pso

import model


def train():
    """
    """

    # Set hyperparameters:
    num_epochs = 10
    swarm_size = 10
    batch_size = 10
    lam = .33
    eta = 1.0


    train_data, train_labels, test_data, test_labels = model.load_data()
    n = len(train_data[0])
    print("n: %d" % n)

    #Save parameters
    params = model.init_params(n)
    vec_params = model.vectorize(params, n)
    dim = len(vec_params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    xopt, fopt = pso(model.get_loss, bounds[0], bounds[1], args=(n, train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)
    print(xopt, fopt)
    params_vec = xopt

    params = model.tensorize(params_vec, n)
    num_correct = 0
    for i in range(len(test_data)):
        dist =  model.get_distribution(params, n, test_data[i])
        if dist[math.floor(test_labels[i])] > 0.5:
            num_correct += 1
    print("test accuracy = ", num_correct/len(test_data))
if __name__ == '__main__':
    train()
