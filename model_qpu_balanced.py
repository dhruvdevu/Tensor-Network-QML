from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ, MEASURE, SWAP
from pyquil.api import WavefunctionSimulator
from pyquil.quil import DefGate
from collections import deque
import numpy as np
import math
import scipy
import time
import data

gates = [[0,1], [2,3], [1,3], [4,5], [6,7], [5,7], [8,9], [10,11], [9, 11], [12,13], [14,15], [13,15], [3,7], [11,15], [7, 15]]

class Model:

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.qc = get_qc("16q-qvm")#get_qc("Aspen-1-15Q-A", as_qvm=True, noisy=False) or get_qc("Aspen-1-16Q-A")
        self.num_trials = kwargs['num_trials']
        if ('seed' in kwargs.keys()):
            seed = kwargs['seed']
            # print(seed)
            np.random.seed(seed)
        self.count = 16*len(gates)# + 6
        self.num_runs = 0
        # self.classes = kwargs['classes']



    def prep_state_program(self, sample):
        #TODO: Prep a state given classical vector x
        prog = Program()
        # print(sample)
        num1 = max(sample)
        num0 = min(sample)
        for i in range(0, self.n):
            # val = (sample[i]-num0)/(num1-num0)
            val = sample[i]
            angle = math.pi*val#/2
            prog.inst(RY(angle, i))
        return prog

    def single_qubit_unitary(self, params, qubit):
        # Get the Quil definition for the new gate
        real_params = params[0:3]
        imag_params = params[3:6]
        real_half = np.array([real_params[0:2], [real_params[1], real_params[2]]])
        imag_half = np.array([imag_params[0:2], [imag_params[1], imag_params[2]]])
        real_matrix = real_half + real_half.T
        real_matrix = real_matrix*0.5*np.eye(2)
        imag_matrix = imag_half - imag_half.T
        hermitian_matrix = real_matrix + 1j*imag_matrix
        unitary_matrix = scipy.linalg.expm(1j*hermitian_matrix)
        # gate_definition = DefGate("GATE", unitary_matrix)
        # Get the gate constructor
        # GATE = gate_definition.get_constructor()
        return unitary_matrix


    def two_qubit_unitary(self, params):
        real_params = params[0:10]
        imag_params = params[10:16]
        real_half = np.array([real_params[0:4], np.concatenate([np.zeros(1), real_params[4:7]]), np.concatenate([[0., 0.], real_params[7:9]]), np.concatenate([[0., 0., 0.], [real_params[9]]])])
        real_matrix = real_half + real_half.T
        real_matrix = real_matrix*0.5*np.eye(4)
        imag_half = np.array([np.concatenate([np.zeros(1), imag_params[0:3]]), np.concatenate([[0., 0.], imag_params[3:5]]), [0., 0., 0. ,imag_params[5]], np.zeros(4)])
        imag_matrix = imag_half - imag_half.T
        hermitian_matrix = real_matrix + 1j*imag_matrix
        unitary_matrix = scipy.linalg.expm(1j*hermitian_matrix)
        return unitary_matrix
        # Get the Quil definition for the new gate
        # gate_definition = DefGate("GATE", unitary_matrix)
        # # Get the gate constructor
        # GATE = gate_definition.get_constructor()
        # return GATE



    def prep_circuit_program(self, params):

        prog = Program()
        for i in range(len(gates)):
            q1 = gates[i][0]
            q2 = gates[i][1]
            index = i*16
            unitary = self.two_qubit_unitary(params[index:index+16])
            gate_definition = DefGate("GATE"+str(q1)+"-"+str(q2), unitary)
            gate = gate_definition.get_constructor()
            prog.inst(gate_definition, gate(q1, q2))

        # index = 20*len(gates)
        # unitary = self.single_qubit_unitary(params[index:index+6], self.n - 1)
        # gate_definition = DefGate("GATE15", unitary)
        # gate = gate_definition.get_constructor()
        # prog.inst(gate_definition, gate(self.n-1))

        return prog

    # def get_params(self, prog):
    #     #18 angles for each 2 qubit gate, 3 more for the last single qubit gate
    #     params = prog.declare('params', memory_type='REAL', memory_size=self.count)
    #     return params

    def get_distribution(self, params, sample):

        start_time = time.time()
        res = self.qc.run_and_measure(self.prep_state_program(sample) + self.prep_circuit_program(params), trials=self.num_trials)[15]
        end_time = time.time()
        print("Time taken for", self.num_trials, " trials =", end_time-start_time)
        self.num_runs += self.num_trials
        count_0 = 0.0
        for x in res:
            if x == 0:
                count_0 += 1.0
        return [count_0/self.num_trials, 1-count_0/self.num_trials]






    def get_loss(self, params, *args, **kwargs):
        loss = 0.0
        samples, labels, lam, eta, batch_size = args
        # We insert some code to only grab some of the training set (randomly drawn, with replacement):
        # print(labels.shape)
        # print(samples.shape)
        # print(samples[1])
        # perm = np.random.permutation(samples.shape[0])
        # samples = samples[perm[:batch_size]]
        # labels = labels[perm[:batch_size]]

        # print(samples.shape, labels.shape)

        for i in range(batch_size):
            sample = samples[i].flatten()
            label = 1
            if labels[i][0] == 1:
                label = 0

            dist = self.get_distribution(params, sample)
            loss += (max(dist[1-label] - dist[label] + lam, 0.0)) ** eta

        loss /= batch_size
        # print("Loss: %f" % loss)
        return loss

    def get_multiloss(self, params_vecs, *args):
        return [self.get_loss(x, args) for x in params_vecs]
