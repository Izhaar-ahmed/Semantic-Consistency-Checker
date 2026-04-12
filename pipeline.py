"""End-to-end consistency-checking pipeline combining bi-encoder and cross-encoder."""

import nltk

from biencoder import BiEncoder
from crossencoder import NLICrossEncoder
from config import SIMILARITY_THRESHOLD, CONTRADICTION_THRESHOLD

nltk.download("punkt_tab", quiet=True)


class ConsistencyPipeline:
    """Detect semantic contradictions in a document or sentence pair."""

    def __init__(self, model_path: str = None):
        """Initialise bi-encoder and cross-encoder components."""
        self._biencoder = BiEncoder()
        if model_path:
            self._crossencoder = NLICrossEncoder(model_path=model_path)
        else:
            self._crossencoder = NLICrossEncoder()

    def check_document(self, text: str) -> dict:
        """Analyse a full document and return consistency score, contradictions, and stats."""
        sentences = nltk.sent_tokenize(text)
        n = len(sentences)
        total_pairs = n * (n - 1) // 2

        candidates = self._biencoder.get_candidate_pairs(sentences, SIMILARITY_THRESHOLD)
        filtered_pairs = len(candidates)
        reduction = 1.0 - (filtered_pairs / max(total_pairs, 1))

        pairs_text = [(sentences[i], sentences[j]) for i, j, _ in candidates]
        verdicts = self._crossencoder.predict_batch(pairs_text) if pairs_text else []

        contradictions = []
        for (i, j, cos), scores in zip(candidates, verdicts):
            if scores["contradiction"] >= CONTRADICTION_THRESHOLD:
                contradictions.append({
                    "sentence1": sentences[i],
                    "sentence2": sentences[j],
                    "confidence": scores["contradiction"],
                    "indices": (i, j),
                })

        checked = max(len(verdicts), 1)
        consistency_score = 1.0 - (len(contradictions) / checked)

        return {
            "consistency_score": round(consistency_score, 4),
            "contradictions": contradictions,
            "stats": {
                "total_pairs": total_pairs,
                "filtered_pairs": filtered_pairs,
                "filter_reduction_pct": round(reduction * 100, 2),
            },
        }

    def check_pair(self, s1: str, s2: str) -> dict:
        """Classify a single sentence pair with NLI scores."""
        scores = self._crossencoder.predict_pair(s1, s2)
        best_label = max(scores, key=scores.get)
        return {
            "label": best_label,
            "confidence": scores[best_label],
            "all_scores": scores,
        }
