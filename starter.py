from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ
from pyquil.parameters import Parameter
from pyquil.parametric import ParametricProgram
from collections import deque
import numpy as np
import math


def prep_state_program(x):
    #TODO: Prep a state given classical vector x
    prog = Program()
    for i in range(0, len(x)):
        angle = math.pi*x[i]/2
        prog.inst(RY(angle, i))
    return prog

def single_qubit_unitary(angles):
    return RX(angles[0]), RZ(angles[1]), RX(angles[2])


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
                single_qubit_gates += [[single_qubit_unitary(params[i][j][k][0]), single_qubit_unitary(params[i][j][k][1])]]
            layer_gates += [single_qubit_gates]
        gates += [layer_gates]
    gates += [single_qubit_unitary(params[math.floor(math.log(n, 2))])]
    return gates

def init_params(n):
    params = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        i_level = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            j_level = []
            for k in range(0, 3):
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
            print(q1, q2)
            for k in range(0, 3):
                prog.inst([g(q1) for g in single_qubit_unitaries[i][j][k][0]])
                prog.inst([g(q2) for g in single_qubit_unitaries[i][j][k][1]])
                prog.inst(CZ(q1, q2))
    prog.inst([g(n - 1) for g in single_qubit_unitaries[math.floor(math.log(n, 2))]])
    prog.measure(n - 1, 0)
            #create each block
    return prog

def vectorize(params, n):
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
    p = Program().inst(prep_state_program(sample) + prep_circuit(n, params))
    qvm = QVMConnection()
    res = qvm.run(p, [0], trials = 100)
    count_0 = 0.0
    for x in res:
        if x[0] == 0:
            count_0 += 1.0
    return (count_0/100, 1-count_0/100)

def get_loss(params_vec, n, sample, label, lam, eta):
    label = math.floor(label)
    dist = get_distribution(tensorize(params_vec, n), n, sample)
    return (max(dist[1 - label] - dist[label] + lam, 0.0)) ** eta



def train():
    data = np.loadtxt(open("data/data.csv", "rb"), delimiter = ",")
    labels = np .loadtxt(open("data/labels.csv", "rb"), delimiter = ",")
    #prep_state_program([7.476498897658023779e-01,2.523501102341976221e-01,0.000000000000000000e+00,0.000000000000000000e+00])
    #Number of qubits
    n = len(data[0])

    #Prepare circuit
    #TODO: Can this be optimized into 1 parametric program?
    # par_prep_state = ParametricProgram(prep_state_program)
    # par_pred_circuit = ParametricProgram(prep_circuit)

    #Shuffle data
    combined = np.hstack((data, labels[np.newaxis].T))
    np.random.shuffle(combined)
    data = combined[:,list(range(n))]
    labels = combined[:,n]
    #Save parameters
    params = init_params(n)
    lam = 0
    eta = 1
    print(get_loss(vectorize(params, n), n, data[0], labels[0], lam, eta))

if __name__ == '__main__':
    train()
