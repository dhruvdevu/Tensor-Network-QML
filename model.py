from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ

from collections import deque
import numpy as np
import math
import scipy

qc = get_qc("4q-qvm")

def prep_state_program(x):
    #TODO: Prep a state given classical vector x
    prog = Program()
    for i in range(0, len(x)):
        angle = math.pi*x[i]/2
        prog.inst(RY(angle, i))
    return prog

def single_qubit_unitary(angles, qubit):
    return RX(angles[0], qubit), RZ(angles[1], qubit), RX(angles[2], qubit)


def prep_parametric_gates(n, params):
    #n is the number of qubits. Here we create a circuit using only using 1 and 2 qubit unitaries
    #Every 1 qubit unitary can be encoded as 3 parametric rotations - rx,rz,rx
    gates = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        layer_gates = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            single_qubit_gates = []
            q1 = 2**i - 1 + j*(2**(i + 1))
            q2 = q1  + 2**i
            for k in range(0, 3):
                single_qubit_gates += [[single_qubit_unitary(params[i][j][k][0], q1), single_qubit_unitary(params[i][j][k][1], q2)]]
            layer_gates += [single_qubit_gates]
        gates += [layer_gates]
    gates += [single_qubit_unitary(params[math.floor(math.log(n, 2))], n - 1)]
    return gates

def init_params(n):
    params = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        i_level = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            j_level = []
            for k in range(0, 3):
                # TODO: Can make program parametric here
                qubit_one_params = 2*math.pi*np.random.rand(3)
                qubit_two_params = 2*math.pi*np.random.rand(3)
                k_level = [list(qubit_one_params), list(qubit_two_params)]
                j_level += [k_level]
            i_level += [j_level]
        params += [i_level]
    params += [list(2*math.pi*np.random.rand(3))]
    return params



def prep_circuit(n, params):
    prog = Program()
    #Prepare parametric gates
    single_qubit_unitaries = prep_parametric_gates(n, params)
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        #number of gates in the ith layer is 2^(log(n)-i-1)
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            q1 = 2**i - 1 + j*(2**(i + 1))
            q2 = q1  + 2**i
            for k in range(0, 3):
                #TODO: call the function to create gates in here instead of creating them beforehand
                #Declare the parameters here and just pass them to the single qqubit unitary function
                prog.inst([g for g in single_qubit_unitaries[i][j][k][0]])
                prog.inst([g for g in single_qubit_unitaries[i][j][k][1]])
                prog.inst(CZ(q1, q2))
    prog.inst([g for g in single_qubit_unitaries[math.floor(math.log(n, 2))]])

            #create each block
    return prog

def vectorize(params, n):
    #TODO: Validate the input
    params_vec = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        i_level = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            j_level = []
            for k in range(0, 3):
                params_vec += params[i][j][k][0]
                params_vec += params[i][j][k][1]
    params_vec += params[math.floor(math.log(n, 2))]
    return params_vec

def tensorize(params_vec, n):
    #TODO: Validate the input
    params_vec = deque(params_vec)
    params = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        i_level = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            j_level = []
            for k in range(0, 3):
                qubit_one_params = []
                for l in range(0, 3):
                    qubit_one_params.append(params_vec.popleft())
                qubit_two_params = []
                for l in range(0, 3):
                    qubit_two_params.append(params_vec.popleft())
                k_level = [qubit_one_params, qubit_two_params]
                j_level += [k_level]
            i_level += [j_level]
        params += [i_level]
    last_unitary = []
    for l in range(0, 3):
        last_unitary += [params_vec.popleft()]
    params.append(last_unitary)
    return params

def get_distribution(params, n, sample):
    # The length of the sample vector should be equal to the number
    # of qubits.
    assert (n == len(sample))
    num_trials = 25
    p = Program().inst(prep_state_program(sample) + prep_circuit(n, params))
    # Only want the nth qubit
    res = qc.run_and_measure(p, trials=num_trials)[n - 1]
    count_0 = 0.0
    # print(res)
    for x in res:
        if x == 0:
            count_0 += 1.0
    return (count_0/num_trials, 1-count_0/num_trials)

def get_loss(params_vec, *args, **kwargs):
    loss = 0.0
    n, samples, labels, lam, eta, batch_size = args
    print("Vec", params_vec)
    params = tensorize(params_vec, n)

    # We insert some code to only grab some of the training set (randomly drawn, with replacement):
    if ('seed' in kwargs.keys()):
        seed = kwargs['seed']
        # print(seed)
        np.random.seed(seed)

    combined = np.hstack((samples, labels[np.newaxis].T))
    np.random.shuffle(combined)
    samples = combined[:,list(range(n))]
    labels = combined[:,n]

    for i in range(batch_size):
        sample = samples[i]
        # print(sample)
        label = math.floor(labels[i])
        dist = get_distribution(params, n, sample)
        loss += (max(dist[1 - label] - dist[label] + lam, 0.0)) ** eta

    loss /= batch_size
    print("Loss: %f" % loss)
    return loss

def get_multiloss(params_vecs, *args):
    return [get_loss(x, args) for x in params_vecs]


def load_data():
    """This method loads the data and separates it into testing and training data."""

    data = np.loadtxt(open("data/data_bw.csv", "rb"), delimiter = ",")
    labels = np .loadtxt(open("data/labels_bw.csv", "rb"), delimiter = ",")
    #prep_state_program([7.476498897658023779e-01,2.523501102341976221e-01,0.000000000000000000e+00,0.000000000000000000e+00])
    #Number of qubits
    n = len(data[0])
    print("n: %d" % n)

    #Prepare circuit
    #TODO: Can this be optimized into 1 parametric program?
    # par_prep_state = ParametricProgram(prep_state_program)
    # par_pred_circuit = ParametricProgram(prep_circuit)

    #Shuffle data
    combined = np.hstack((data, labels[np.newaxis].T))
    np.random.shuffle(combined)
    data = combined[:,list(range(n))]
    labels = combined[:,n]
    num_training = math.floor(0.7*len(data))
    train_data = data[:num_training]
    train_labels = labels[:num_training]
    test_data = data[num_training:]
    test_labels = labels[num_training:]

    return (train_data, train_labels, test_data, test_labels)
