from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

#Simulating Markoivan amplitude damping on a qubit that is prepared in initial state|1>
#The environment is in ground state |0>


#Amplitude damping parameter
gamma = 0.3

#Gamma expressed as rotation angle
theta = 2 * np.arcsin(np.sqrt(gamma))


#Quantum circuit with one system qubit, one environmental qubit
# and one classical register for measurement
sys = QuantumRegister(1,name='S')
env = QuantumRegister(1,name='E')

out = ClassicalRegister(1,name='out')


qc = QuantumCircuit(sys,env,out)

#Preparing the system qubit to state |1>
qc.x(0)

#Interaction between system and environment
qc.cry(theta,0,1)

qc.cx(1,0)
#This changes system state to |0> if the environment gets exited

qc.barrier()

#Measure the system
qc.measure(0,0)

#Visualize the circuit
qc.draw('mpl')
plt.show()

#Simulate the quantum circuit and plot the states of the system
sim = AerSimulator()
result = sim.run(qc, shots=1000).result()
counts = result.get_counts()

print(counts)

plot_histogram(counts)
plt.show()


