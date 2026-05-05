from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

#Choosing the least busy IBM quantum computer for these simulations

service = QiskitRuntimeService()
backend = service.least_busy(
    operational=True, simulator=False, min_num_qubits=127
)

pm = generate_preset_pass_manager(backend=backend)


#Parameter used in the simulations
n_steps = 15
shots = 2000

T1 = 50e-6
dt = 2e-6

gamma_step = 1 - np.exp(-dt / T1)
theta = 2 * np.arcsin(np.sqrt(gamma_step))

#Function that build the non-Markovian cirucit
def build_circuit_non_markovian(step):
    qc = QuantumCircuit(2, 1)

    qc.h(0)

    for _ in range(step):
        qc.cry(theta, 0, 1)
        qc.cx(1, 0)
        qc.rx(0.4, 1)

    qc.measure(0, 0)
    return qc

#Define needed simulators

ideal_sim = AerSimulator()
noisy_sim = AerSimulator.from_backend(backend)


#We can store the probabilities of the different simulations to these

p_ideal = []
p_noisy = []
p_real = []
times_ideal = []
times_noisy = []
times_real = []


for step in range(n_steps + 1):
    qc_ideal = build_circuit_non_markovian(step)

    ideal_counts = ideal_sim.run(qc_ideal, shots=shots).result().get_counts()

    
    p_ideal.append(ideal_counts.get('1', 0) / shots)

    times_ideal.append(step * dt * 1e6)  # µs


#Now simulate hardware noise using a simulated quantum computer

for step in range(n_steps+1):
    qc_noisy = build_circuit_non_markovian(step)
    
    qc_noisy = transpile(qc_noisy, noisy_sim)
    
    noisy_counts = noisy_sim.run(qc_noisy, shots=shots).result().get_counts()
    
    p_noisy.append(noisy_counts.get('1', 0) / shots)
    
    times_noisy.append(step * dt * 1e6)


#Compare these to a real quantum computer

circuits = []

for step in range(n_steps+1):
    
    qc_real = build_circuit_non_markovian(step)
    
    isa_qc = pm.run(qc_real)
    
    circuits.append(isa_qc)

sampler = Sampler(mode=backend)

job = sampler.run(circuits, shots=shots)

result = job.result()

for i, pub_result in enumerate(result):
    
    counts_real = pub_result.data.c.get_counts()
    
    p_real.append(counts_real.get('1', 0) / shots)
    
    times_real.append(i * dt * 1e6)


#We can obtain the status of the job and other possibly interesting metrics

job_id = job.job_id()


print(f"Job ID: {job_id}")

job = service.job(job_id)
print(f"Job status: {job.status()}")

metrics = job.metrics()
quantum_seconds = metrics['usage']['quantum_seconds']
print(f"The QPU was used for: {metrics['usage']['quantum_seconds']} seconds")


plt.figure(figsize=(7,5))
plt.plot(times_ideal, p_ideal, 'o-', label='Simulation')
plt.plot(times_noisy,p_noisy, 'o-', label='Noisy')
plt.plot(times_real,p_real, 'o-', label=f"Real ({backend.name})")

plt.xlabel("Time [µs]")
plt.ylabel("Probability: P(|1⟩)")
plt.title("Non-Markovian amplitude damping: Initial state |+⟩")
plt.grid()
plt.legend()
plt.show()