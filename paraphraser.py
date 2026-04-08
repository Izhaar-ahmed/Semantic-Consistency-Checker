"""T5-based paraphrase generator with caching for training-time augmentation."""

from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

from config import PARAPHRASE_MODEL, DEVICE, NUM_PARAPHRASES, MAX_SEQ_LEN


class Paraphraser:
    """Generate paraphrases using a fine-tuned T5 model."""

    def __init__(self):
        """Initialise T5 model, tokenizer, and an in-memory cache."""
        self._tokenizer = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL)
        self._model = T5ForConditionalGeneration.from_pretrained(PARAPHRASE_MODEL).to(DEVICE)
        self._cache: dict[str, list[str]] = {}

    def generate(self, sentence: str, n: int = NUM_PARAPHRASES) -> list[str]:
        """Return *n* paraphrases for a single sentence."""
        if sentence in self._cache:
            return self._cache[sentence][:n]

        text = f"paraphrase: {sentence}"
        encoding = self._tokenizer(
            text, max_length=MAX_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(DEVICE)

        outputs = self._model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=MAX_SEQ_LEN,
            num_beams=n + 2,
            num_return_sequences=n,
            early_stopping=True,
        )

        paraphrases = [
            self._tokenizer.decode(o, skip_special_tokens=True) for o in outputs
        ]
        self._cache[sentence] = paraphrases
        return paraphrases[:n]

    def augment(self, sentences: list[str]) -> list[tuple[str, str]]:
        """Create (original, paraphrase) positive pairs for every sentence."""
        pairs: list[tuple[str, str]] = []
        for sent in tqdm(sentences, desc="Paraphrasing", unit="sent"):
            for para in self.generate(sent):
                pairs.append((sent, para))
        return pairs
