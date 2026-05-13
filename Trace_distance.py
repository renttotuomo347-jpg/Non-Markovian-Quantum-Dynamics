from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
import numpy as np
import matplotlib.pyplot as plt


#Define c1(R,t)
def c1(R, t):

    if R < 0.5:
        c1 = np.exp(-t/2.0) * (
            np.cosh(t*np.sqrt(1.0-2.0*R)/2.0)
            + 1.0/np.sqrt(1.0-2.0*R)
            * np.sinh(t*np.sqrt(1.0-2.0*R)/2.0)
        )

    else:
        c1 = np.exp(-t/2.0) * (
            np.cos(t*np.sqrt(2.0*R-1.0)/2.0)
            + 1.0/np.sqrt(2.0*R-1.0)
            * np.sin(t*np.sqrt(2.0*R-1.0)/2.0)
        )

    return c1


def amplitude_damping(R, t, initial_state='+'):

    qc = QuantumCircuit(2)

    sys = 0
    env = 1

    #Prepare the two initial state that we use for trace distance
    #Initial state |+⟩
    if initial_state == '+':
        qc.h(sys)

    #Initial state |-⟩
    elif initial_state == '-':
        qc.x(sys)
        qc.h(sys)
        
    #Damping parameter
    theta = np.arccos(c1(R, t))

    #Entaglement and qubit swap
    qc.cu(2*theta, 0, 0, 0, sys, env)
    qc.cx(env, sys)

    #No measurement

    return qc

#We need to make our quantum circuit into a density matrix
def density_matrix(qc):

    rho = DensityMatrix.from_instruction(qc)

    #Trace out the environment qubit
    rho_sys = partial_trace(rho, [1])

    return rho_sys



#Calculate trace distance between states |+⟩ and |-⟩
def trace_distance(rho1, rho2):
    
    #ρ₁ - ρ₂
    delta = rho1.data - rho2.data
    
    #Eigenvalues
    eigvals = np.linalg.eigvals(delta)

    return 0.5 * np.sum(np.abs(eigvals))



R_values = [0.2, 2.0, 20]

times = np.linspace(0, 20, 100)


#Calculate trace distance for different values of R
for R in R_values:
    
    distances = []


    for t in times:

        qc_plus = amplitude_damping(R, t, '+')
        qc_minus = amplitude_damping(R, t, '-')

        rho_plus = density_matrix(qc_plus)
        rho_minus = density_matrix(qc_minus)

        D = trace_distance(rho_plus, rho_minus)

        distances.append(D)

    #Plot the results
    plt.figure(figsize=(7,5))

    plt.plot(times, distances, '-', label=f"Trace distance: R = {R}")
    plt.xlabel("Time")
    plt.ylabel("Trace Distance")
    plt.title("Trace Distance Dynamics: ρ₁=|+⟩ and ρ₂=|-⟩")
    plt.legend()
    plt.grid()
    plt.show()








