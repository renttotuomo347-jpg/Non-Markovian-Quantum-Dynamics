from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Pauli
import numpy as np
import matplotlib.pyplot as plt


#Again define c1(R,t)
def c1(R, t):

    if R < 0.5:
        val = np.exp(-t/2.0) * (
            np.cosh(t*np.sqrt(1.0 - 2.0*R)/2.0)
            +
            (1.0/np.sqrt(1.0 - 2.0*R))
            *
            np.sinh(t*np.sqrt(1.0 - 2.0*R)/2.0)
        )

    else:
        val = np.exp(-t/2.0) * (
            np.cos(t*np.sqrt(2.0*R - 1.0)/2.0)
            +
            (1.0/np.sqrt(2.0*R - 1.0))
            *
            np.sin(t*np.sqrt(2.0*R - 1.0)/2.0)
        )

    return val



#Define amplitude damping
def amplitude_damping(R, t, initial_state='+'):

    qc = QuantumCircuit(2)

    sys = 0
    env = 1

    #Choose initial states as |+⟩ and |-⟩
    if initial_state == '+':

        qc.h(sys)

    theta = np.arccos(c1(R, t))

    qc.cry(2*theta, sys, env)
    qc.cx(env, sys)

    #No measurement
    
    return qc



#Define density matrix for quantum circuits
def density_matrix(qc):

    rho = DensityMatrix.from_instruction(qc)

    # Trace out environment qubit
    rho_sys = partial_trace(rho, [1])

    return rho_sys




#Calculate Bloch vector components which are just expectation values
# of the corresponding Pauli matrices
def bloch_components(rho):

    x = np.real(rho.expectation_value(Pauli("X")))
    y = np.real(rho.expectation_value(Pauli("Y")))
    z = np.real(rho.expectation_value(Pauli("Z")))

    return x, y, z



R_values = [0.2, 2.0, 20]

times = np.linspace(0, 20, 100)


#Calculate expectation values for all values of R
for R in R_values:
    #Store the values here
    x_vals_plus = []
    y_vals_plus = []
    z_vals_plus = []

    #x_vals_minus = []
    #y_vals_minus = []
    #_vals_minus = []

    #Time evolution
    for t in times:

        #Prepare quantum circuit for state |+⟩
        qc_plus = amplitude_damping(R, t, '+')
        
        #Prepare quantum circuit for state |-⟩
        #qc_minus = amplitude_damping(R, t, '-')
        
        #Density matrix
        rho_plus = density_matrix(qc_plus)
        #rho_minus = density_matrix(qc_minus)
        
        # Bloch components
        x1, y1, z1 = bloch_components(rho_plus)
        
        #x2, y2, z2 = bloch_components(rho_minus)

        #Add the values to storage
        x_vals_plus.append(x1)
        y_vals_plus.append(y1)
        z_vals_plus.append(z1)

        #x_vals_minus.append(x2)
        #y_vals_minus.append(y2)
        #z_vals_minus.append(z2)




    #Plot the dynamics of Bloch vector components
    plt.figure(figsize=(8,5))

    plt.plot(times, x_vals_plus, label='⟨X⟩')
    plt.plot(times, y_vals_plus, label='⟨Y⟩')
    plt.plot(times, z_vals_plus, label='⟨Z⟩')
    
    #plt.plot(times, x_vals_minus, label='⟨X⟩: |-⟩')
    #plt.plot(times, y_vals_minus, label='⟨Y⟩: |-⟩')
    #plt.plot(times, z_vals_minus, label='⟨Z⟩: |-⟩')

    plt.xlabel("Time")
    plt.ylabel("Bloch Components for intial state |+⟩")

    plt.title(
        f"Bloch Vector Dynamics (R={R})"
    )

    plt.legend()
    plt.grid()

    plt.show()



