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

def prep_parametric_gates(n, params):
    #n is the number of qubits. Here we create a circuit using only using 1 and 2 qubit unitaries
    #Every 1 qubit unitary can be encoded as 3 parametric rotations - rx,rz,rx
    prog = Program()
    for i in range(0, math.log(n, 2)):
        j = 0
        #number of gates in the ith layer is 2^(n-i-1)
        layer = []
        angles = params[i]
        for j in range(0, math.pow(2, n-i-1)):
            thetas = relevant angles
            single_qubit_unitaries = make them (use angles and params)
            q1 = 2**i - 1 + j*(2**(i + 1))
            q2 = 2**i - 1 + j*(2**(i + 1) + )
            for k in range(0, 3):
                prog.inst(single_qubit_unitaries[k](q1))
                prog.inst(single_qubit_unitaries[k+1](q2))
                prog.inst(CZ(q1, q2))
            #create each block

    p = Program()
    p.inst()

def train():
    data = np.loadtxt(open("data/data.csv", "rb"), delimiter = ",")
    labels = np.loadtxt(open("data/labels.csv", "rb"), delimiter = ",")
    prep_state_program([7.476498897658023779e-01,2.523501102341976221e-01,0.000000000000000000e+00,0.000000000000000000e+00])

    par_p = ParametricProgram(prep_parametric_program)
    #Prepare parametric gates
    #Run SPSA
    #Save parameters

    #define qubits in each layer



if __name__ == "__main__":
    train()
