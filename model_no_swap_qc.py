from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ, MEASURE
from collections import deque
import numpy as np
import math
import scipy
import time

gates = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [8,9], [9,10], [10, 11], [12,13], [13,14], [14,15], [3,7], [11,15], [7, 15]]

class Model:

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.qc = get_qc("Aspen-1-16Q-A")#get_qc("Aspen-1-15Q-A", as_qvm=True, noisy=False)
        self.num_trials = kwargs['num_trials']
        if ('seed' in kwargs.keys()):
            seed = kwargs['seed']
            # print(seed)
            np.random.seed(seed)
        self.count = 18*len(gates) + 3
        prog = self.prep_state_program()
        self.program = self.prep_circuit_program(prog)
        self.program.wrap_in_numshots_loop(shots=self.num_trials)
        compile_start = time.time()
        self.executable = self.qc.compile(self.program)
        compile_end = time.time()
        print("Compile time = ", compile_end - compile_start)
        self.num_runs = 0



    def prep_state_program(self):
        #TODO: Prep a state given classical vector x
        prog = Program()
        sample = prog.declare('sample', memory_type='REAL', memory_size=self.n)
        for i in range(0, self.n):
            angle = math.pi*sample[i]/2
            prog.inst(RY(angle, i))
        return prog

    def single_qubit_unitary(self, angles, qubit):
        return RX(angles[0], qubit), RZ(angles[1], qubit), RX(angles[2], qubit)

    def prep_circuit_program(self, prog):
        #Prepare parametric gates
        ro = prog.declare('ro', memory_type='BIT')
        self.params = self.get_params(prog)
        params_dict = {}
        for i in range(len(gates)):
            q1 = gates[i][0]
            q2 = gates[i][1]
            for k in range(0, 3):
                index = i*18+6*k
                angles = self.params[index], self.params[index+1], self.params[index+2]
                prog.inst([g for g in self.single_qubit_unitary(angles, q1)])
                index += 3
                angles = self.params[index], self.params[index+1], self.params[index+2]
                prog.inst([g for g in self.single_qubit_unitary(angles, q2)])
                prog.inst(CZ(q1, q2))

        index = 18*len(gates)
        angles = self.params[index], self.params[index+1], self.params[index+2]
        prog.inst([g for g in self.single_qubit_unitary(angles, self.n - 1)])

        prog += MEASURE(15, ro)
        return prog

    def get_params(self, prog):
        #18 angles for each 2 qubit gate, 3 more for the last single qubit gate
        params = prog.declare('params', memory_type='REAL', memory_size=self.count)
        return params

    def get_distribution(self, params, sample):
        # The length of the sample vector should be equal to the number
        # of qubits.
        assert (self.n == len(sample))
        # Only want the nth qubit
        res = self.qc.run(self.executable, memory_map={'params': params, 'sample': sample})
        self.num_runs += self.num_trials
        count_0 = 0.0
        for x in res:
            if x == 0:
                count_0 += 1.0
        return (count_0/self.num_trials, 1-count_0/self.num_trials)

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
