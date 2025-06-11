# src/eval/snn_metrics.py

import numpy as np
import json
import os

def evaluate_spikes(path, save_path="outputs/snn_metrics.json", compute_entropy=True):
    """
    Evaluate spike train simulation output and save metrics as JSON.

    Parameters:
    - path: str, path to .npz file from Brian2 simulation
    - save_path: str, where to save the output JSON metrics
    - compute_entropy: bool, whether to calculate neuron entropy

    Returns:
    - metrics: dict, dictionary of calculated metrics
    """
    data = np.load(path)
    spike_indices = data["spike_indices"]
    spike_times = data["spike_times"]

    n_neurons = int(spike_indices.max()) + 1 if spike_indices.size > 0 else 0
    total_time_ms = spike_times.max() if spike_times.size > 0 else 1

    # Calculate spike counts per neuron
    spike_counts = np.bincount(spike_indices, minlength=n_neurons)
    firing_rates = spike_counts / total_time_ms * 1000  # spikes/sec

    # Calculate spike density
    spike_density = len(spike_indices) / (n_neurons * total_time_ms)

    metrics = {
        "total_spikes": int(len(spike_indices)),
        "n_neurons": int(n_neurons),
        "duration_ms": float(total_time_ms),
        "avg_firing_rate_hz": float(np.mean(firing_rates)) if firing_rates.size > 0 else 0,
        "spike_density": float(spike_density),
        "max_firing_rate": float(np.max(firing_rates)) if firing_rates.size > 0 else 0,
        "min_firing_rate": float(np.min(firing_rates)) if firing_rates.size > 0 else 0,
    }

    if compute_entropy and firing_rates.sum() > 0:
        prob = firing_rates / firing_rates.sum()
        entropy = -np.sum(prob * np.log2(prob + 1e-9))
        metrics["neuron_entropy_bits"] = float(entropy)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Spike metrics saved to: {save_path}")
    return metrics