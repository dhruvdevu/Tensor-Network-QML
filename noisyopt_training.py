import numpy as np
import math
from noisyopt import minimizeSPSA

import model


def train():
    """
    """

    # Set hyperparameters:
    batch_size = 10
    lam = .33
    eta = 1.0
    num_iterations = 40
    a = .2
    c = .2


    train_data, train_labels, test_data, test_labels = model.load_data()
    n = len(train_data[0])
    print("n: %d" % n)

    #Save parameters
    params = model.init_params(n)
    vec_params = model.vectorize(params, n)
    dim = len(vec_params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    args=(n, train_data, train_labels, lam, eta, batch_size)
    result = minimizeSPSA(model.get_loss, args=args, x0=vec_params, paired=True, niter=num_iterations, a=a, c=c)#, bound=bounds)

#    xopt, fopt = pso(model.get_loss, bounds[0], bounds[1], args=(n, train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)'
    print(result)
    params_vec = result.x

    params = model.tensorize(params_vec, n)
    num_correct = 0
    for i in range(len(test_data)):
        dist =  model.get_distribution(params, n, test_data[i])
        if dist[math.floor(test_labels[i])] > 0.5:
            num_correct += 1
    print("test accuracy = ", num_correct/len(test_data))
if __name__ == '__main__':
    train()
