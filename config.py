"""Central configuration for the Semantic Consistency pipeline."""

import torch

# ── Model identifiers ──────────────────────────────────────────────
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-base"
PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"

# ── Thresholds ──────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.40
CONTRADICTION_THRESHOLD = 0.70

# ── Training / inference settings ───────────────────────────────────
BATCH_SIZE = 32
MAX_SEQ_LEN = 128
def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()
SAVE_DIR = "models/"

# ── Dataset names ───────────────────────────────────────────────────
STSB_DATASET = "sentence-transformers/stsb"
SNLI_DATASET = "stanfordnlp/snli"
MNLI_DATASET = "nyu-mll/multi_nli"

# ── Paraphrase generation ──────────────────────────────────────────
NUM_PARAPHRASES = 2
