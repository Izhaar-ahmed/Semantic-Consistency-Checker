"""Fine-tune the NLI cross-encoder on SNLI + MNLI + paraphrase-augmented data."""

import os

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

from config import CROSS_ENCODER_MODEL, BATCH_SIZE, SAVE_DIR, DEVICE
from data import load_nli
from paraphraser import Paraphraser


def _nli_to_examples(df, limit: int = 20_000) -> list[InputExample]:
    """Convert NLI DataFrame rows into InputExample objects."""
    examples = []
    for _, row in df.head(limit).iterrows():
        examples.append(InputExample(texts=[row["premise"], row["hypothesis"]], label=int(row["label"])))
    return examples


def _paraphrase_to_examples(paraphraser: Paraphraser, sentences: list[str], limit: int = 200) -> list[InputExample]:
    """Generate augmented positive (entailment=0) pairs via paraphrasing."""
    pairs = paraphraser.augment(sentences[:limit])
    return [InputExample(texts=[s1, s2], label=0) for s1, s2 in pairs]


def fine_tune_crossencoder() -> None:
    """Train cross-encoder on combined NLI + paraphrase data and save to disk."""
    print("[1/4] Loading NLI data …")
    nli_df = load_nli()
    nli_examples = _nli_to_examples(nli_df)

    print("[2/4] Generating paraphrase augmentations …")
    paraphraser = Paraphraser()
    unique_sents = nli_df["premise"].drop_duplicates().tolist()
    para_examples = _paraphrase_to_examples(paraphraser, unique_sents)

    all_examples = nli_examples + para_examples
    print(f"[3/4] Training on {len(all_examples)} examples …")

    # DeBERTa-v3 runs faster on CPU than MPS; force CPU for cross-encoder training
    model = CrossEncoder(CROSS_ENCODER_MODEL, num_labels=3, device="cpu")
    loader = DataLoader(all_examples, shuffle=True, batch_size=BATCH_SIZE)
    model.old_fit(
        train_dataloader=loader,
        epochs=1,
        warmup_steps=int(0.1 * len(loader)),
        show_progress_bar=True,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "nli-crossencoder")
    model.save(save_path)
    print(f"[4/4] Model saved → {save_path}")


if __name__ == "__main__":
    fine_tune_crossencoder()
