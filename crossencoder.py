"""NLI cross-encoder for entailment / neutral / contradiction classification."""

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import CROSS_ENCODER_MODEL, CONTRADICTION_THRESHOLD, BATCH_SIZE, DEVICE, MAX_SEQ_LEN


class NLICrossEncoder:
    """Predict NLI labels for sentence pairs using DeBERTa-v3."""

    def __init__(self, model_path: str = CROSS_ENCODER_MODEL):
        """Load tokenizer and classification model."""
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        self._model.eval()
        self._labels = [self._model.config.id2label[i] for i in range(self._model.config.num_labels)]

    def predict_pair(self, s1: str, s2: str) -> dict[str, float]:
        """Return {entailment, neutral, contradiction} probabilities for one pair."""
        return self.predict_batch([(s1, s2)])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Return NLI probability dicts for a batch of pairs."""
        results: list[dict[str, float]] = []
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start: start + BATCH_SIZE]
            s1s, s2s = zip(*batch)
            enc = self._tokenizer(
                list(s1s), list(s2s), padding=True, truncation=True,
                max_length=MAX_SEQ_LEN, return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                results.append({label: float(row[i]) for i, label in enumerate(self._labels)})
        return results

    def is_contradiction(self, s1: str, s2: str) -> bool:
        """Return True if the pair is classified as a contradiction."""
        scores = self.predict_pair(s1, s2)
        return scores["contradiction"] >= CONTRADICTION_THRESHOLD
