from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ

from collections import deque
import numpy as np
import math
import scipy
import time

qc = get_qc("4q-qvm")
prog = Program()
angles =
prog.inst(RY(angle, 1))
