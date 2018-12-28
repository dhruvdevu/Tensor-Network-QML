import numpy as np
import math
from pyswarm import pso
import utils
import model


def train(train_data, train_labels):
    """
    """

    # Set hyperparameters:
    num_epochs = 20
    swarm_size = 10
    batch_size = 10
    lam = .33
    eta = 1.0


    n = len(train_data[0])
    # print("n: %d" % n)

    #Save parameters
    mod = model.Model(n=n, num_trials=100)
    params = list(2*math.pi*np.random.rand(mod.count))
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    xopt, fopt = pso(mod.get_loss, bounds[0], bounds[1], args=(train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)
    print(xopt, fopt)
    params = xopt
    return params, mod
