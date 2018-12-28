import numpy as np
import math
from noisyopt import minimizeSPSA

import model_refactored as model
import numpy as np
import utils

def train():
    """
    """

    # Set hyperparameters:
    batch_size = 10
    lam = .33
    eta = 1.0
    num_iterations = 100 #40
    a = .2
    c = .2


    train_data, train_labels, test_data, test_labels = utils.load_data(BLACK_AND_WHITE=False)
    n = len(train_data[0])
    print("n: %d" % n)

    #Save parameters
    mod = model.Model(n=n, num_trials=100)
    params = list(2*math.pi*np.random.rand(mod.count))
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    args = (train_data, train_labels, lam, eta, batch_size)
    result = minimizeSPSA(mod.get_loss, args=args, x0=params, paired=True, niter=num_iterations, a=a, c=c)#, bound=bounds)

#    xopt, fopt = pso(model.get_loss, bounds[0], bounds[1], args=(n, train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)'
    print(result)
    params = result.x
 #    params = [4.5118702 , 1.93607608, 0.15151715, 5.33794621, 0.64906416,
 # 2.88375913, 6.31813019, 4.44123923, 1.3385396 , 6.23507886,
 # 5.66210074, 0.29622917, 4.08113191, 5.7846549 , 0.95420014,
 # 3.35288529, 4.88335534, 4.94185989, 5.10633353, 2.64940975,
 # 1.78174675, 0.68132855, 0.99488483, 1.30540764, 2.3492256 ,
 # 2.48538058, 1.29176267, 5.96417216, 5.78923658, 4.52804206,
 # 3.59387521, 0.58382647, 3.97880432, 5.64226554, 3.50279262,
 # 2.28137216, 0.5724503 , 4.26512265, 2.31448848, 2.07738777,
 # 2.37836655, 3.19155593, 1.37552593, 2.08623775, 1.32121266,
 # 3.95739933, 4.48050696, 4.5196555 , 1.05309731, 1.52623498,
 # 5.39814951, 0.18820205, 5.5738594 , 0.55618373, 1.58210923,
 # 0.7405427 , 2.76271768]
    np.savetxt("params/noisy_params_grey_v0.csv", params, delimiter = ",")
    num_correct = 0
    for i in range(len(test_data)):
        dist =  mod.get_distribution(params, test_data[i])
        if dist[math.floor(test_labels[i])] > 0.5:
            num_correct += 1
    print("test accuracy = ", num_correct/len(test_data))
    print("number of circuit runs = ", mod.num_runs)
if __name__ == '__main__':
    train()
