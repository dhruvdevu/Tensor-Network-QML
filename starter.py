from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import CNOT, H, RZ
import numpy as np
import math


def prep_state_program(x):
    #TODO: Prep a state given classical vector x
    prog = Program()
    for i in range(0, len(x)):
        angle = math.pi*x[i]/2
        prog.inst(RZ(angle, i))
    print(prog)
    return prog

def train():
    data = 
    prep_state_program([7.476498897658023779e-01,2.523501102341976221e-01,0.000000000000000000e+00,0.000000000000000000e+00])



if __name__ == "__main__":
    train()
