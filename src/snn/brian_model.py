# 🧠 NeuroGenAI | Universal Brian2 SNN Engine

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os

def run_brian2_simulation(
    spike_matrix: np.ndarray,
    duration_ms: int = 100,
    neuron_eqs: str = "dv/dt = (1.0 - v) / (10*ms) : 1",
    threshold: str = "v > 0.9",
    reset: str = "v = 0",
    method: str = "exact",
    syn_weight: float = 0.2,
    stdp: bool = False,
    plot_path: str = "outputs/snn_spike_plot.png",
    save_path: str = "outputs/snn_sim_results.npz",
    monitor_neurons: int = 10
):
    # 🔄 Reset Brian2 state
    start_scope()
    defaultclock.dt = 1 * ms
    duration = duration_ms * ms

    # 🧬 Convert binary spike matrix → spike events
    n_neurons = spike_matrix.shape[1]
    input_indices, input_times = np.nonzero(spike_matrix)
    spike_times = input_indices * ms
    spike_neurons = input_times
    spike_gen = SpikeGeneratorGroup(n_neurons, spike_neurons, spike_times)

    # 🧠 Neuron model
    neurons = NeuronGroup(n_neurons, neuron_eqs, threshold=threshold, reset=reset, method=method)
    neurons.v = 0

    # 🔗 Synapse model
    syn = Synapses(spike_gen, neurons, on_pre=f'v_post += {syn_weight}')
    syn.connect(j='i')

    # 🧪 Optional STDP plasticity
    if stdp:
        stdp_eqs = '''
            w : 1
            dApre/dt = -Apre / (20*ms) : 1 (event-driven)
            dApost/dt = -Apost / (20*ms) : 1 (event-driven)
        '''
        stdp_syn = Synapses(neurons, neurons, model=stdp_eqs,
                            on_pre='''
                                v_post += w
                                Apre += 0.01
                                w = clip(w + Apost, 0, 1)
                            ''',
                            on_post='''
                                Apost += 0.01
                                w = clip(w + Apre, 0, 1)
                            ''')
        stdp_syn.connect(condition='i != j', p=0.1)
        stdp_syn.w = '0.2 + 0.1*rand()'

    # 🧲 Monitors
    M = StateMonitor(neurons, 'v', record=True)
    spikes = SpikeMonitor(neurons)

    print(f"🚀 Running simulation for {duration_ms} ms with {n_neurons} neurons...")
    run(duration)

    # 📊 Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Spike Raster")
    plt.plot(spikes.t / ms, spikes.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')

    plt.subplot(2, 1, 2)
    plt.title("Membrane Potentials")
    for i in range(min(monitor_neurons, len(M.v))):
        plt.plot(M.t / ms, M.v[i], label=f"Neuron {i}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (v)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"📁 Saved raster & voltages to: {plot_path}")

    # 💾 Save simulation data
    np.savez(save_path,
             spike_times=spikes.t / ms,
             spike_indices=spikes.i,
             voltages=M.v[:])
    print(f"✅ Simulation data saved to {save_path}")

    return M, spikes, neurons