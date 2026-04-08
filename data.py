"""Data loading and label-mapping utilities for STS-B, NLI, and paraphrase datasets."""

import pandas as pd
from datasets import load_dataset

from config import STSB_DATASET, SNLI_DATASET, MNLI_DATASET


def map_sts_to_label(score: float) -> str:
    """Map a 0-5 STS score to a categorical consistency label."""
    if score >= 3.5:
        return "consistent"
    if score <= 2.0:
        return "contradiction"
    return "neutral"


def load_stsb() -> dict[str, pd.DataFrame]:
    """Load STS-B splits and return {train, validation, test} DataFrames."""
    ds = load_dataset(STSB_DATASET)
    frames = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        df = df.rename(columns={"score": "score"})
        df["label"] = df["score"].apply(map_sts_to_label)
        frames[split] = df[["sentence1", "sentence2", "score", "label"]]
    return frames


def load_nli(snli_limit: int = 50_000, mnli_limit: int = 30_000) -> pd.DataFrame:
    """Load SNLI + MNLI training splits (sampled) into one combined DataFrame."""
    snli = load_dataset(SNLI_DATASET, split="train").to_pandas()
    mnli = load_dataset(MNLI_DATASET, split="train").to_pandas()

    snli = snli[["premise", "hypothesis", "label"]]
    mnli = mnli[["premise", "hypothesis", "label"]]

    # Filter invalid labels first, then sample
    snli = snli[snli["label"].isin([0, 1, 2])].sample(n=min(snli_limit, len(snli)), random_state=42)
    mnli = mnli[mnli["label"].isin([0, 1, 2])].sample(n=min(mnli_limit, len(mnli)), random_state=42)

    combined = pd.concat([snli, mnli], ignore_index=True)
    return combined
