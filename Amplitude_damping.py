from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService


#We have to define c1(R,t) for both R < 0,5 and R > 0,5

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



#Lets build the needed quantum circuit to performe the simulation
def amplitude_damping(R, t):
    #Two qubits and one classical bit
    q = QuantumRegister(2, 'q')
    c = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(q, c)

    #System is qubit 0 and environment is qubit 1
    sys = 0
    env = 1

    #Preparation of the initial state |1⟩
    qc.x(sys)

    #The angle of the controlled-U gate depends on the function c1(R,t)
    theta = np.arccos(c1(R, t))

    #CRY gate applies a R_y rotation to environment depending on the control qubit
    #This creates entanglement between the system qubit and environmental qubit
    qc.cry(2*theta, sys, env)
    #Controlled-X gate swaps the target qubit if the control qubit is in state |1⟩
    qc.cx(env, sys)

    #Measurement on the system qubit
    qc.measure(sys, 0)

    return qc


#Running a simulation on a real quantum computer


service = QiskitRuntimeService()
backend = service.least_busy(
    operational=True, simulator=False, min_num_qubits=5
    )


#Parameters used in the simulation
shots = 1000

R_values = [0.2, 2.0, 20]

n_times = 5

  

sampler = Sampler(mode=backend)
#We can store the information in here 
all_circuits = []
labels = []
time_dict = {}
probs = {}

#Run the simulation for all values of R we want to use
for R in R_values:

    t_max = 6.0 * np.pi / np.sqrt(abs(2.0*R - 1.0))
    times = np.linspace(0, t_max, n_times)

    probs[R] = []
    time_dict[R] = times

    for t in times:

        qc = amplitude_damping(R, t)

        qc_t = transpile(
            qc,
            backend=backend,
            optimization_level=3
        )

        all_circuits.append(qc_t)
        labels.append((R, t))

job = sampler.run(all_circuits, shots=shots)
result = job.result()

#Order the results according to R
for i, (R, t) in enumerate(labels):

    pub_result = result[i]
    
    counts = pub_result.data[i].c.get_counts()

    p1 = counts.get('1', 0) / shots

    probs[R].append(p1)

#We can obtain job id and other possibly interesting metrics like used QPU time,
# job status, and used backend

job_id = job.job_id()

print("Using backend: ", backend)

print(f"Job ID: {job_id}")

job = service.job(job_id)
print(f"Job status: {job.status()}")

metrics = job.metrics()
quantum_seconds = metrics['usage']['quantum_seconds']
print(f"The QPU was used for: {metrics['usage']['quantum_seconds']} seconds")

#Plot the result for each R value
for R in R_values:

    times = time_dict[R]

    theory = np.abs(c1(R, times))**2

    plt.plot(times, theory, '-', label=f"Theory R={R}")
    plt.plot(times, probs[R], 'o', label=f"Hardware R={R}")

plt.xlabel("Time")
plt.ylabel("P(|1⟩)")
plt.title("Amplitude Damping: Initial state |1⟩")
plt.grid()
plt.legend()
plt.show()