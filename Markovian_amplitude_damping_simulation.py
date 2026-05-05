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


#Parameters used in the simulations

n_steps = 15
shots = 2000

T1 = 50e-6
dt = 2e-6

gamma_step = 1 - np.exp(-dt / T1)
theta = 2 * np.arcsin(np.sqrt(gamma_step))


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

#Define a function that builds the needed quantum circuit for these simulations

def build_circuit(step):
    qc = QuantumCircuit(step + 1, 1)

    #We can prepare the system in |1⟩ or |+⟩

    #Preparation to |1⟩ by making next line active and making Hadamar into a comment
    #qc.x(0)

    #Preparation to |+>
    qc.h(0) #Make this into a comment when preparing to initial state |1⟩

    #Interactions between the system and the environment
    for i in range(step):
        env = i + 1

        qc.cry(theta, 0, env)
        
        qc.cx(env, 0)
    
    qc.barrier()
    
    qc.measure(0, 0)
    return qc


#Let's performe an ideal simulation without any hardware noise

for step in range(n_steps + 1):

    #Build the circuit with function
    qc_ideal = build_circuit(step)
    
    #Transpile the circuit for the simulation
    qc_ideal = transpile(qc_ideal, ideal_sim)
    
    #Store the counts of the simulation
    ideal_counts = ideal_sim.run(qc_ideal, shots=shots).result().get_counts()
    
    #Choose the number of counts that are in state |1⟩ and divide ny total amount
    # of simulation shots to get the probability of |1⟩
    p_ideal.append(ideal_counts.get('1', 0) / shots)
    
    #Timesteps in microseconds
    times_ideal.append(step * dt * 1e6)


#Now simulate hardware noise using a simulated quantum computer

for step in range(n_steps+1):
    
    #Build the circuit
    qc_noisy = build_circuit(step)
    
    #Transpile the circuit for the simulation
    qc_noisy = transpile(qc_noisy, noisy_sim)
    
    #Store the counts of the simulation
    noisy_counts = noisy_sim.run(qc_noisy, shots=shots).result().get_counts()
    
    #Choose the number of counts that are in state |1⟩ and divide ny total amount
    # of simulation shots to get the probability of |1⟩
    p_noisy.append(noisy_counts.get('1', 0) / shots)
    
    #Timesteps in microseconds
    times_noisy.append(step * dt * 1e6)


#Compare these to a real quantum computer

#For each timestep we build a new quantum circuit and store it here
circuits = []

for step in range(n_steps+1):
    
    #Build the circuit
    qc_real = build_circuit(step)
    
    isa_qc = pm.run(qc_real)
    
    #Store the circuit
    circuits.append(isa_qc)

#Run the simulation
sampler = Sampler(mode=backend)

job = sampler.run(circuits, shots=shots)

result = job.result()

#Store the counts of the simulation
for i, pub_result in enumerate(result):
    
    counts_real = pub_result.data.c.get_counts()
    
    p_real.append(counts_real.get('1', 0) / shots)
    
    times_real.append(i * dt*1e6)


#We can obtain the status of the job and other possibly interesting metrics

job_id = job.job_id()


print(f"Job ID: {job_id}")

job = service.job(job_id)
print(f"Job status: {job.status()}")

metrics = job.metrics()
quantum_seconds = metrics['usage']['quantum_seconds']
print(f"The QPU was used for: {metrics['usage']['quantum_seconds']} seconds")


#We need the ideal theoretical model that is in this case exponential

t = np.linspace(0, n_steps * dt * 1e6, 100)
theory = np.exp(-t * 1e-6 / T1) / 2


#We can plot the different types of simulations
# so we can easily compare them to eachother

plt.figure(figsize=(7,5))

plt.plot(t, theory, '--', label='Exponential fit')
plt.plot(times_ideal, p_ideal, 'o-', label='Simulation')
plt.plot(times_noisy, p_noisy, 'o-', label='Noisy')
plt.plot(times_real, p_real, "-o", label=f"Real ({backend.name})")

plt.xlabel("Time [µs]")
plt.ylabel("Probability: P(|1⟩)")
plt.title("Markovian Amplitude Damping: Initial state |+>")
plt.grid()
plt.legend()
plt.show()
