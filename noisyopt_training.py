import numpy as np
import math
from noisyopt import minimizeSPSA


import numpy as np
import utils

def train(train_data, train_labels, mod, params=None):
    """
    """

    # Set hyperparameters:
    # batch_size = 25
    # lam = .33
    # eta = 1.0
    # num_iterations = 200
    # a = .2
    # c = .2
    batch_size = 25
    lam = .234
    eta = 5.59
    num_iterations = 10
    a = 33.0
    c = 28.0
    alpha = 0.658
    gamma = 4.13


    n = len(train_data[0])
    print("n: %d" % n)

    #Save parameters
    if params is None:
        params = list(2*math.pi*np.random.rand(mod.count))
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    args = (train_data, train_labels, lam, eta, batch_size)
    result = minimizeSPSA(mod.get_loss, args=args, x0=params, paired=True, niter=num_iterations, a=a,alpha=alpha, c=c, gamma=gamma, disp=False, callback=utils.save_params)#, bound=bounds)

#    xopt, fopt = pso(model.get_loss, bounds[0], bounds[1], args=(n, train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)'
    print(result)
    params = result.x
    print("number of training circuit runs = ", mod.num_runs)
    return params, mod
