# 🧠 NeuroGenAI | Spike Encoder: DNABERT to Neuron Firing Rates

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SpikeEncoder:
    def __init__(self, rate_max_hz=100):
        self.rate_max_hz = rate_max_hz
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        normed = self.scaler.fit_transform(embeddings)
        print(f"✅ Normalized embeddings to [0–1] shape: {normed.shape}")
        return normed

    def to_firing_rates(self, normalized: np.ndarray) -> np.ndarray:
        firing_rates = normalized * self.rate_max_hz
        print(f"🌟 Converted to firing rates (Hz), max: {self.rate_max_hz} Hz")
        print("Top 5 firing rates:", np.round(firing_rates[0][:5], 2))
        return firing_rates

    def save_rate_vectors(self, firing_rates: np.ndarray, path="data/processed/firing_rates.npy"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, firing_rates)
        print(f"📁 Saved firing rate vectors to: {path}")

    def encode_and_save(self, embeddings_path: str, out_path: str = "data/processed/firing_rates.npy"):
        embeddings = np.load(embeddings_path)
        normed = self.normalize_embeddings(embeddings)
        rates = self.to_firing_rates(normed)
        self.save_rate_vectors(rates, out_path)
        return rates