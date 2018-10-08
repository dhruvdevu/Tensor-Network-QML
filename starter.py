from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import CNOT, H, RZ
from pyquil.parameters import Parameter
from pyquil.parametric import ParametricProgram
import numpy as np
import math


def prep_state_program(x):
    #TODO: Prep a state given classical vector x
    prog = Program()
    for i in range(0, len(x)):
        angle = math.pi*x[i]/2
        prog.inst(RY(angle, i))
    print(prog)
    return prog

def single_qubit_unitary(angles):
    return RX(angle[0]), RZ(angle[1]), RX(angle[2])


def prep_parametric_gates(n, params):
    #n is the number of qubits. Here we create a circuit using only using 1 and 2 qubit unitaries
    #Every 1 qubit unitary can be encoded as 3 parametric rotations - rx,rz,rx
    gates = []
    for i in range(0, math.log(n, 2)):
        layer_gates = []
        for j in range(0, math.pow(2, n-i-1)):
            single_qubit_gates = []
            q1 = 2**i - 1 + j*(2**(i + 1))
            q2 = 2**i + j*(2**(i + 1) + 1)
            for k in range(0, 3):
                single_qubit_gates += [[single_qubit_unitary(params[i][j][k][0]), single_qubit_unitary(params[i][j][k][0])]]
            layer_gates +=[two_qubit_gate]
        gates += [layer_gates]
    return gates




def prep_circuit(n, params):
    prog = Program()
    single_qubit_unitaries = prep_parametric_gates(n, params)
    for i in range(0, math.log(n, 2)):
        #number of gates in the ith layer is 2^(n-i-1)
        angles = params[i]
        for j in range(0, math.pow(2, n-i-1)):
            q1 = 2**i - 1 + j*(2**(i + 1))
            q2 = 2**i + j*(2**(i + 1) + 1)
            for k in range(0, 3):
                prog.inst([g(q1) for g in single_qubit_unitaries[i][j][k][0]])
                prog.inst([g(q2) for g in single_qubit_unitaries[i][j][k][1]])
                prog.inst(CZ(q1, q2))
        prog.inst(single_qubit_unitary[math.log(n, 2) + 1](n - 1))
        prog.measure(0, 0)
            #create each block
    return prog

def train():
    data = np.loadtxt(open("data/data.csv", "rb"), delimiter = ",")
    labels = np.loadtxt(open("data/labels.csv", "rb"), delimiter = ",")
    prep_state_program([7.476498897658023779e-01,2.523501102341976221e-01,0.000000000000000000e+00,0.000000000000000000e+00])

    #Prepare parametric gates
    #Prepare circuit
    par_p = ParametricProgram(prep_circuit)
    #Prepare parametric gates
    #Run SPSA
    #Save parameterss

    #define qubits in each layer



if __name__ == "__main__":
    train()
