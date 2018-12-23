import numpy as np
import math
from noisyopt import minimizeSPSA

import model_refactored as model


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
    mod = model.Model(n=n, num_trials=25)
    params = list(2*math.pi*np.random.rand(mod.count))
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    args = (train_data, train_labels, lam, eta, batch_size)
    result = minimizeSPSA(mod.get_loss, args=args, x0=params, paired=True, niter=num_iterations, a=a, c=c)#, bound=bounds)

#    xopt, fopt = pso(model.get_loss, bounds[0], bounds[1], args=(n, train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)'
    print(result)
    params = result.x

    num_correct = 0
    for i in range(len(test_data)):
        dist =  mod.get_distribution(params, test_data[i])
        if dist[math.floor(test_labels[i])] > 0.5:
            num_correct += 1
    print("test accuracy = ", num_correct/len(test_data))
if __name__ == '__main__':
    train()
