from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ

from collections import deque
import numpy as np
import math
import scipy
import time

class Model:

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.qc = get_qc("16q-qvm") #get_qc("Aspen-1-16Q-A-qvm")#get_qc("16q-qvm")
        self.num_trials = kwargs['num_trials']
        if ('seed' in kwargs.keys()):
            seed = kwargs['seed']
            # print(seed)
            np.random.seed(seed)
        self.params_mapping = self.prep_parameter_mapping()
        self.count = 0
        self.program = Program().inst(self.prep_state_program() + self.prep_circuit_program())
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

    def prep_parameter_mapping(self):
        #n is the number of qubits. Here we create a circuit using only using 1 and 2 qubit unitaries
        gates = []
        l = math.floor(math.log(self.n, 2))
        for i in range(0, l):
            layer_gates = []
            for j in range(0, math.floor(math.pow(2, l-i-1))):
                single_qubit_gates = []
                for k in range(0, 3):
                    single_qubit_gates += [[-1, -1]]
                layer_gates += [single_qubit_gates]
            gates += [layer_gates]
        gates += [-1]
        return gates

    def init_params(self, num_params):
        params = list(2*math.pi*np.random.rand(num_params))


    def prep_circuit_program(self):
        prog = Program()
        #Prepare parametric gates
        self.params, self.params_mapping = self.get_params_and_mapping(prog)
        params_dict = {}
        l = math.floor(math.log(self.n, 2))
        for i in range(0, l):
            #number of gates in the ith layer is 2^(log(n)-i-1)
            for j in range(0, math.floor(math.pow(2, l-i-1))):
                q1 = 2**i - 1 + j*(2**(i + 1))
                q2 = q1  + 2**i
                for k in range(0, 3):
                    #TODO: call the function to create gates in here instead of creating them beforehand
                    #Declare the parameters here and just pass them to the single qqubit unitary function
                    #need to have the program handle a vector of params since our optimizers cannot handle a tensor or a dictionary
                    #make mapping between params tensor and vector
                    index = self.params_mapping[i][j][k][0]
                    angles = self.params[index], self.params[index+1], self.params[index+2]
                    prog.inst([g for g in self.single_qubit_unitary(angles, q1)])
                    index = self.params_mapping[i][j][k][1]
                    angles = self.params[index], self.params[index+1], self.params[index+2]
                    prog.inst([g for g in self.single_qubit_unitary(angles, q2)])
                    prog.inst(CZ(q1, q2))

        index = self.params_mapping[math.floor(math.log(self.n, 2))]
        angles = self.params[index], self.params[index+1], self.params[index+2]
        prog.inst([g for g in self.single_qubit_unitary(angles, math.floor(math.log(self.n, 2)))])
        ro = prog.declare('ro', memory_type='BIT', memory_size=1)
        prog.measure(self.n - 1, ro[0])
        return prog

    def get_params_and_mapping(self, prog):
        #TODO: Validate the input
        params_mapping = self.params_mapping
        count = 0
        l = math.floor(math.log(self.n, 2))
        for i in range(0, l):
            for j in range(0, math.floor(math.pow(2, l-i-1))):
                for k in range(0, 3):
                    params_mapping[i][j][k][0] = count
                    count += 3
                    params_mapping[i][j][k][1] = count
                    count += 3
        params_mapping[math.floor(math.log(self.n, 2))] = count
        count += 3
        self.count = count
        print(count)
        params = prog.declare('params', memory_type='REAL', memory_size=count) #TODO: Fix this!!!
        return params, params_mapping


    def get_distribution(self, params, sample):
        # The length of the sample vector should be equal to the number
        # of qubits.
        assert (self.n == len(sample))
        # Only want the nth qubit
        res = self.qc.run(self.executable, memory_map={'params': params, 'sample': sample})
        self.num_runs += self.num_trials
        count_0 = 0.0
        # print(res)
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
