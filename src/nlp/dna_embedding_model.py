import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

class DNAEmbedder:
    def __init__(self, model_id="armheb/DNA_bert_6", k=6, device=None):
        self.model_id = model_id
        self.k = k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🧠 Loading model {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def tokenize(self, sequence):
        sequence = sequence.upper().replace(" ", "").replace("\n", "")
        kmers = [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]
        return kmers

    def embed(self, sequence, max_tokens=512):
        kmers = self.tokenize(sequence)
        if len(kmers) == 0:
            print("⚠️ Sequence too short for k-mer embedding.")
            return np.zeros(768)

        chunks = [kmers[i:i + max_tokens] for i in range(0, len(kmers), max_tokens)]
        embeddings = []

        for chunk in chunks:
            input_text = " ".join(chunk)
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model(**inputs)
                emb = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(emb)

        return np.mean(embeddings, axis=0)

    def embed_batch(self, sequences):
        all_vecs = []
        for idx, seq in enumerate(sequences):
            print(f"🔁 Embedding sequence {idx+1}/{len(sequences)}...")
            vec = self.embed(seq)
            all_vecs.append(vec)
        return np.vstack(all_vecs)