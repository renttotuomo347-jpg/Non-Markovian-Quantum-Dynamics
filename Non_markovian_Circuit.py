from qiskit import QuantumCircuit
import numpy as np

#Non-Markovian amplitude damping

#Parameters
n_steps = 5
shots = 2000

T1 = 50e-6
dt = 2e-6

gamma_step = 1 - np.exp(-dt / T1)
theta = 2 * np.arcsin(np.sqrt(gamma_step))

#Function that builds the quantum cirucit
def build_circuit_non_markovian(step):
    qc = QuantumCircuit(2, 1)

    qc.h(0)

    for _ in range(step):
        qc.cry(theta, 0, 1)
        qc.cx(1, 0)
        qc.rx(0.4, 1)

    qc.measure(0, 0)
    return qc

qc = build_circuit_non_markovian(n_steps)

#Visualize the circuit
qc.draw('mpl')

