"""SBERT bi-encoder for fast candidate-pair filtering via cosine similarity."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import SBERT_MODEL, SIMILARITY_THRESHOLD, BATCH_SIZE, DEVICE


class BiEncoder:
    """Encode sentences and retrieve high-similarity candidate pairs."""

    def __init__(self):
        """Load the SBERT model onto the configured device."""
        self._model = SentenceTransformer(SBERT_MODEL, device=DEVICE)

    def encode(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences into 768-dim normalised embeddings."""
        embeddings = self._model.encode(
            sentences, batch_size=BATCH_SIZE, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return embeddings

    def get_candidate_pairs(
        self, sentences: list[str], threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[tuple[int, int, float]]:
        """Return (i, j, cosine) for all pairs above the similarity threshold."""
        embeddings = self.encode(sentences)
        n = len(sentences)
        sim_matrix = embeddings @ embeddings.T
        pairs: list[tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    pairs.append((i, j, score))
        return pairs
