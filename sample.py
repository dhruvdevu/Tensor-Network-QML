# from pyquil import Program, get_qc
# from pyquil.gates import CNOT, H, RZ, RY, RX, CZ
#
# import numpy as np
#
#
# qc = get_qc("4q-qvm")
# prog = Program()
# theta = prog.declare('theta', memory_type='REAL', memory_size=1)
# ro = prog.declare('ro', memory_type='BIT', memory_size=1)
# prog.inst(RX(theta, 0))
# prog.measure(0, ro[0])
# prog.wrap_in_numshots_loop(shots=10)
# exec = qc.compile(prog)
# res = qc.run(exec, memory_map={'theta': [np.pi]})
# print(res)

import matplotlib.pyplot as plt
import utils
train_data, train_labels, test_data, test_labels = utils.load_data('mnist')
print(train_labels[1])
print(train_data[1])
plt.imshow(train_data[0])
