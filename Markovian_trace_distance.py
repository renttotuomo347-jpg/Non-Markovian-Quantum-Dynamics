from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import numpy as np
import matplotlib.pyplot as plt

#Parameters used in the calculations
n_steps = 15
dt = 2e-6
T1 = 50e-6

gamma_step = 1 - np.exp(-dt / T1)
theta = 2 * np.arcsin(np.sqrt(gamma_step))

#Define a function that builds the needed quantum circuit
def build_circuit(step, init_state):
    qc = QuantumCircuit(step + 1)

    #Prepare the two initial states
    
    #Initial state |+⟩
    if init_state == "+":
        qc.h(0)
    
    #Initial state |-⟩
    elif init_state == "-":
        qc.x(0)
        qc.h(0)

    #Create a new environmental qubit for each interaction
    for i in range(step):
        env = i + 1
        qc.cry(theta, 0, env)
        qc.cx(env, 0)
        
    #We don't measure the system
    return qc

#Define a function that calculates the trace distance of two
# different initial states
def trace_distance(rho1, rho2):
    #Calculate the differences of the two states
    diff = rho1.data - rho2.data

    #Eigenvalues of the difference
    eigvals = np.linalg.eigvals(diff)
    
    #Return half of the absolute value of the difference
    return 0.5 * np.sum(np.abs(eigvals))


#Simulate the Markovian amplitude damping of the states

#Store the results in these
times = []
distances = []

for step in range(n_steps + 1):

    # two different initial states
    qc1 = build_circuit(step, "+")
    qc2 = build_circuit(step, "-")

    state1 = Statevector.from_instruction(qc1)
    state2 = Statevector.from_instruction(qc2)

    if step == 0:
        rho1 = DensityMatrix(state1)
        rho2 = DensityMatrix(state2)
    else:
        rho1 = partial_trace(state1, list(range(1, step + 1)))
        rho2 = partial_trace(state2, list(range(1, step + 1)))

    # compute trace distance
    D = trace_distance(rho1, rho2)

    distances.append(D)
    
    #Timesteps in microseconds
    times.append(step * dt * 1e6)

#Plot the trace distance of the states
plt.figure(figsize=(6,4))
plt.plot(times, distances, 'o-')
plt.xlabel("Time [µs]")
plt.ylabel("Trace distance D(ρ₁, ρ₂)")
plt.title("Markovian trace distance decay: ρ₁=|+⟩ and ρ₂=|-⟩")
plt.grid()
plt.show()