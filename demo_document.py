import time
import nltk
from biencoder import BiEncoder
from crossencoder import NLICrossEncoder
from pipeline import ConsistencyPipeline
from config import SIMILARITY_THRESHOLD, CONTRADICTION_THRESHOLD

def main():
    # 30-sentence document combining the user's provided sentences with thematic extensions
    document = (
        "The weather was extremely cold. "
        "John wore a heavy winter jacket. "
        "The temperature was below freezing. "
        "John was dressed in light summer clothes. " # Contradicts 2
        "Sarah made hot soup for dinner. "
        "The roads were icy and dangerous. "
        "It was a warm sunny afternoon. " # Contradicts 1, 3
        "Children played outside in the snow. "
        "The forecast predicted heavy rainfall. "
        "Birds were singing in the warm breeze. " # Contradicts 1
        "Sarah served a refreshing cold salad for dinner. " # Contradicts 5
        "The lake was completely frozen over. "
        "People were ice skating on the lake. "
        "A boat was sailing swiftly across the open lake water. " # Contradicts 12
        "The heating system in the house broke down. "
        "Everyone stayed warm near the fireplace. "
        "The house was comfortably heated by the central HVAC system. " # Contradicts 15
        "John shivered as he stepped outside. "
        "Sweat dripped down John's face from the intense heat. " # Contradicts 18
        "The snow plows worked all night to clear the streets. "
        "The streets were completely dry and dusty. " # Contradicts 20, 6
        "Mary built a snowman in the front yard. "
        "The harsh winter storm caused power outages. "
        "All the city lights shone brightly throughout the night. " # Contradicts 23
        "The trees were bare without any leaves. "
        "Lush green foliage covered the tall trees. " # Contradicts 25
        "The local school was closed due to the blizzard. "
        "Students attended all their classes normally. " # Contradicts 27
        "The fireplace crackled safely in the living room. "
        "The living room was entirely empty and completely freezing cold." # Contradicts 29, 16
    )

    sentences = nltk.sent_tokenize(document)
    n = len(sentences)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sentences[i], sentences[j]))

    print(f"Total sentences: {n}")
    print(f"Total pairs to evaluate: {len(pairs)}\n")

    biencoder = BiEncoder()
    crossencoder = NLICrossEncoder()
    pipeline = ConsistencyPipeline()

    # Method 1: SBERT Only
    print("Running Method 1: SBERT only...")
    start = time.time()
    embeddings = biencoder.encode(sentences)
    _ = embeddings @ embeddings.T
    sbert_time = time.time() - start

    # Method 2: CrossEncoder Only
    print("Running Method 2: CrossEncoder only...")
    start = time.time()
    _ = crossencoder.predict_batch(pairs)
    ce_time = time.time() - start

    # Method 3: Pipeline (SBERT Filter + CrossEncoder)
    print("Running Method 3: Our Pipeline (SBERT filter + CrossEncoder)...")
    start = time.time()
    res = pipeline.check_document(document)
    pipe_time = time.time() - start

    print("\n── Performance Comparison ───────────────────")
    print(f"SBERT Only:         {sbert_time:.4f} seconds (Creates embeddings, but lacks contradiction verification)")
    print(f"CrossEncoder Only:  {ce_time:.4f} seconds (Verifies all {len(pairs)} pairs)")
    print(f"Our Pipeline:       {pipe_time:.4f} seconds (Verifies only relevant filtered pairs)")

    print(f"\nPipeline efficiently reduced the search space!")
    print(f"Filtered {res['stats']['total_pairs']} raw pairs down to just {res['stats']['filtered_pairs']} candidate pairs (Reduced by {res['stats']['filter_reduction_pct']}%).\n")

    print(f"── Contradictions Found ({len(res['contradictions'])}) ───────────────")
    for idx, c in enumerate(res['contradictions'], 1):
        print(f"[{idx}] S1: {c['sentence1']}")
        print(f"    S2: {c['sentence2']}")
        print(f"    Confidence: {c['confidence']:.4f}\n")

if __name__ == "__main__":
    main()
