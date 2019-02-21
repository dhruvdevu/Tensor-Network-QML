from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ, MEASURE
from pyquil.api import WavefunctionSimulator
from collections import deque
import numpy as np
import math
import scipy
import time

wf_sim = WavefunctionSimulator()
gates = [[0,1], [2,3], [1,3], [4,5], [6,7], [5,7], [8,9], [10,11], [9, 11], [12,13], [14,15], [13,15], [3,7], [11,15], [7, 15]]

class Model:

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.qc = get_qc("16q-qvm")#get_qc("Aspen-1-15Q-A", as_qvm=True, noisy=False)
        self.num_trials = kwargs['num_trials']
        if ('seed' in kwargs.keys()):
            seed = kwargs['seed']
            # print(seed)
            np.random.seed(seed)
        self.count = 18*len(gates) + 3
        self.num_runs = 0



    def prep_state_program(self, sample):
        #TODO: Prep a state given classical vector x
        prog = Program()
        for i in range(0, self.n):
            angle = math.pi*sample[i]#/2
            prog.inst(RY(angle, i))
        return prog

    def single_qubit_unitary(self, angles, qubit):
        return RX(angles[0], qubit), RZ(angles[1], qubit), RX(angles[2], qubit)

    def prep_circuit_program(self, params):

        prog = Program()
        for i in range(len(gates)):
            q1 = gates[i][0]
            q2 = gates[i][1]
            for k in range(0, 3):
                index = i*18+6*k
                angles = params[index], params[index+1], params[index+2]
                prog.inst([g for g in self.single_qubit_unitary(angles, q1)])
                index += 3
                angles = params[index], params[index+1], params[index+2]
                prog.inst([g for g in self.single_qubit_unitary(angles, q2)])
                prog.inst(CZ(q1, q2))

        index = 18*len(gates)
        angles = params[index], params[index+1], params[index+2]
        prog.inst([g for g in self.single_qubit_unitary(angles, self.n - 1)])

        return prog

    def get_params(self, prog):
        #18 angles for each 2 qubit gate, 3 more for the last single qubit gate
        params = prog.declare('params', memory_type='REAL', memory_size=self.count)
        return params

    def get_distribution(self, params, sample):
        # The length of the sample vector should be equal to the number
        # of qubits.
        assert (self.n == len(sample))

        start_time = time.time()
        self.num_runs += 1
        wave_func = wf_sim.wavefunction(self.prep_state_program(sample) + self.prep_circuit_program(params))
        prob_dict = wave_func.get_outcome_probs()
        prob0 = 0.0
        for bitstring, prob in prob_dict.items():
            #15th qubit
            if bitstring[0] == '0':
                prob0 += prob
        # print(prob0)
        end_time  = time.time()
        # print("Time to get distribution = ", end_time - start_time)
        return (prob0, 1-prob0)


    def get_loss(self, params, *args, **kwargs):
        loss = 0.0
        samples, labels, lam, eta, batch_size = args
        # We insert some code to only grab some of the training set (randomly drawn, with replacement):

        combined = np.hstack((samples, labels[np.newaxis].T))
        np.random.shuffle(combined)
        samples = combined[:,list(range(self.n))]
        labels = combined[:,self.n]

        for i in range(batch_size):
            sample = samples[i]
            # print(sample)
            label = math.floor(labels[i])
            dist = self.get_distribution(params, sample)
            loss += (max(dist[1 - label] - dist[label] + lam, 0.0)) ** eta

        loss /= batch_size
        print("Loss: %f" % loss)
        return loss

    def get_multiloss(self, params_vecs, *args):
        return [self.get_loss(x, args) for x in params_vecs]
