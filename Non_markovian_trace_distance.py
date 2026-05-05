from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_steps = 15

T1 = 50e-6
dt = 2e-6

gamma_step = 1 - np.exp(-dt / T1)
theta = 2 * np.arcsin(np.sqrt(gamma_step))


#Function that builds the quantum cirucit
def build_circuit_non_markovian(step, init_state):
    qc = QuantumCircuit(2, 1)

    #Prepare sthe two initial states
    if init_state == "plus": #|+⟩
        qc.h(0)
    elif init_state == "minus": #|-⟩
        qc.x(0)
        qc.h(0)
        
    #
    for _ in range(step):
        qc.cry(theta, 0, 1)
        qc.cx(1, 0)
        qc.rx(0.4, 1)

    return qc


#Function that calculates the trace distance
def trace_distance(rho1, rho2):
    
    diff = rho1.data - rho2.data
    eigvals = np.linalg.eigvals(diff)
    return 0.5 * np.sum(np.abs(eigvals))


#Simultate the system to calculate the trace distance decay
times = []
distances = []

#Timesteps
for step in range(n_steps + 1):

    qc1 = build_circuit_non_markovian(step, "plus")
    qc2 = build_circuit_non_markovian(step, "minus")

    state1 = Statevector.from_instruction(qc1)
    state2 = Statevector.from_instruction(qc2)

    #In non-Markovian case we only have one environmental qubit that 
    # interacts multiple times with the system
    env_qubit = [1]

    rho1 = partial_trace(state1, env_qubit)
    rho2 = partial_trace(state2, env_qubit)

    D = trace_distance(rho1, rho2)

    distances.append(D)
    times.append(step * dt * 1e6)  # µs


# --- plot ---
plt.figure(figsize=(7,5))
plt.plot(times, distances, 'o-', label="Trace distance")

plt.xlabel("Time [µs]")
plt.ylabel("Trace distance")
plt.title("Non-Markovian trace distance decay: ρ₁=|+⟩ and ρ₂=|-⟩")
plt.grid()
plt.legend()
plt.show()