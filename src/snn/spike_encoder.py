import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class SpikeEncoder:
    def __init__(self, rate_max_hz=100, duration_ms=500, dt_ms=1.0, stdp_ready=False, seed=42):
        self.rate_max_hz = rate_max_hz
        self.duration_ms = duration_ms
        self.dt_ms = dt_ms
        self.stdp_ready = stdp_ready
        self.seed = seed
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        np.random.seed(self.seed)

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        normed = self.scaler.fit_transform(embeddings)
        print(f"✅ Normalized embeddings to [0–1], shape: {normed.shape}")
        return normed

    def to_firing_rates(self, normalized: np.ndarray) -> np.ndarray:
        rates = normalized * self.rate_max_hz
        print(f"🌡️ Scaled to firing rates (Hz), max rate: {self.rate_max_hz} Hz")
        print("🔝 Top 5 firing rates (first sample):", np.round(rates[0][:5], 2))
        return rates

    def encode_and_save(self, embeddings: np.ndarray, output_dir="data/processed", prefix="spikes"):

        normed = self.normalize_embeddings(embeddings)
        firing_rates = self.to_firing_rates(normed)

        out_path = os.path.join(output_dir, f"{prefix}_rate_vectors.npy")
        np.save(out_path, firing_rates)

        print(f"📁 Saved firing rates to: {out_path}")
        return firing_rates

    def generate_poisson_spike_train(self, firing_rates: np.ndarray, save_path="data/processed/spike_train.npy"):
        """
        Input: firing_rates [n_samples, n_features]
        Output: spike_train [timesteps, n_neurons]
        """
        timesteps = int(self.duration_ms / self.dt_ms)
        n_neurons = firing_rates.shape[1]

        spike_trains = []
        print(f"📏 Generating Poisson spikes: duration {self.duration_ms}ms, dt {self.dt_ms}ms")

        for sample_idx, rate_vec in enumerate(firing_rates):
            spike_prob = (rate_vec / 1000.0) * self.dt_ms  # Convert Hz to prob
            spikes = np.random.rand(timesteps, n_neurons) < spike_prob
            spike_trains.append(spikes.astype(int))

        # Optionally: average if many samples, or just take first for visual
        spike_train = spike_trains[0]
        np.save(save_path, spike_train)
        print(f"💾 Saved spike train to: {save_path}")
        return spike_train

    def plot_raster(self, spike_train: np.ndarray, save_path="outputs/spike_train_preview.png"):

        plt.figure(figsize=(10, 5))
        t, n = spike_train.shape
        for neuron in range(n):
            times = np.where(spike_train[:, neuron])[0]
            plt.scatter(times, [neuron]*len(times), s=1)

        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.title("Spike Raster Plot")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"🖼️ Spike raster saved to: {save_path}")