from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os

# 📥 Load spike train
data_dir = "data/processed"
spike_train_path = os.path.join(data_dir, "spike_train.npy")
spike_matrix = np.load(spike_train_path)  # shape: [timesteps, neurons]
print("✅ Loaded spike matrix:", spike_matrix.shape)

def run_brian2_simulation(
    spike_path="data/processed/spike_train.npy",
    duration_ms=100,
    stdp=False,
    plot_path="outputs/snn_spike_plot.png",
    save_path="outputs/snn_sim_results.npz"
):
    spike_matrix = np.load(spike_path)
    print("✅ Loaded spike matrix:", spike_matrix.shape)

    M, spikes, neurons = run_brian_simulation(spike_matrix, duration_ms=duration_ms, stdp=stdp)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.title("🧠 Spike Raster")
    plt.plot(spikes.t/ms, spikes.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')

    plt.subplot(2,1,2)
    plt.title("🔋 Membrane Potentials")
    for i in range(min(10, len(M.v))):
        plt.plot(M.t/ms, M.v[i], label=f"Neuron {i}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (v)")

    plt.tight_layout()
    plt.savefig(plot_path)
    print("📁 Saved raster & voltages to:", plot_path)

    np.savez(save_path, spike_times=spikes.t/ms, spike_indices=spikes.i, voltages=M.v[:])
    print(f"✅ Simulation data saved to {save_path}")

# ⚙️ Brian2 Parameters
def run_brian_simulation(spike_matrix, duration_ms=100, stdp=False):
    start_scope()  # 🔥 Resets Brian2 to clean state
    defaultclock.dt = 1*ms
    duration = duration_ms * ms
    
    n_neurons = spike_matrix.shape[1]
    n_steps = spike_matrix.shape[0]

    input_indices, input_times = np.nonzero(spike_matrix)
    spike_times = input_indices * ms
    spike_neurons = input_times

    spike_gen = SpikeGeneratorGroup(n_neurons, spike_neurons, spike_times)

    eqs = '''
    dv/dt = (1.0 - v) / (10*ms) : 1
    '''
    neurons = NeuronGroup(n_neurons, eqs, threshold='v > 0.9', reset='v = 0', method='exact')
    neurons.v = 0

    syn = Synapses(spike_gen, neurons, on_pre='v_post += 0.2')
    syn.connect(j='i')

    if stdp:
        syn_stdp = Synapses(neurons, neurons,
                            model='''w : 1
                                     dApre/dt = -Apre / (20*ms) : 1 (event-driven)
                                     dApost/dt = -Apost / (20*ms) : 1 (event-driven)''',
                            on_pre='''v_post += w
                                      Apre += 0.01
                                      w = clip(w + Apost, 0, 1)''',
                            on_post='''Apost += 0.01
                                       w = clip(w + Apre, 0, 1)''')
        syn_stdp.connect(condition='i!=j', p=0.1)
        syn_stdp.w = '0.2 + 0.1*rand()'

    M = StateMonitor(neurons, 'v', record=True)
    spikes = SpikeMonitor(neurons)

    print("🚀 Running simulation for", duration_ms, "ms...")
    net = Network(collect())
    net.run(duration)

    return M, spikes, neurons

# 🧪 Run
duration_ms = 100
M, spikes, neurons = run_brian_simulation(spike_matrix, duration_ms=duration_ms)

# 📊 Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.title("Spike Raster")
plt.plot(spikes.t/ms, spikes.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')

plt.subplot(2,1,2)
plt.title("Membrane Potentials")
for i in range(min(10, len(M.v))):
    plt.plot(M.t/ms, M.v[i], label=f"Neuron {i}")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (v)")

plt.tight_layout()
plot_path = "outputs/snn_spike_plot.png"
plt.savefig(plot_path)
print("📁 Saved raster & voltages to:", plot_path)

# 💾 Save spike state
np.savez("outputs/snn_sim_results.npz", spike_times=spikes.t/ms, spike_indices=spikes.i, voltages=M.v[:])
print("✅ Simulation data saved to outputs/snn_sim_results.npz")