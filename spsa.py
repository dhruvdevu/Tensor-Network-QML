import numpy as np
import math
import numpy as np
import utils

def train(train_data, train_labels, mod, params=None):
    """
    """

    batch_size = 25
    lam = .234
    eta = 5.59
    num_iterations = 500
    a = 28.0
    b = 33.0
    A = 74.1
    gamma = 0.882
    t = 0.658
    s = 4.13

    n = len(train_data[0])
    print("n: %d" % n)

    #Save parameters
    print(params)
    if params is None:
        params = 2*math.pi*np.random.rand(mod.count)
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    v = np.zeros(params.shape)
    for k in range(num_iterations):
        #Good choice for Delta is the Radechemar distribution acc to wiki
        delta = np.random.binomial(n=1, p=0.5, size=dim)
        alpha = a/(k+1+A)**s
        beta = b/(k+1)**t
        perturb = (params + alpha*delta)%(2*np.pi)
        L1 = mod.get_loss(perturb, train_data, train_labels, lam, eta, batch_size)
        perturb = (params - alpha*delta)%(2*np.pi)
        L2 = mod.get_loss(perturb, train_data, train_labels, lam, eta, batch_size)
        g = (L1-L2)/(2*alpha)
        v = gamma*v - g*beta*delta
        params = (params + v)%(2*np.pi)
        utils.save_params(params)


    print(params)

    print("number of training circuit runs = ", mod.num_runs)
    return params, mod
