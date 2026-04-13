import sys
import nltk
from pipeline import ConsistencyPipeline
from config import SIMILARITY_THRESHOLD, CONTRADICTION_THRESHOLD

def main():
    print("Loading pipeline models... (This may take a moment)")
    pipeline = ConsistencyPipeline()
    # Expose underlying encoders for the manual 2-sentence detailed trace
    biencoder = pipeline._biencoder
    crossencoder = pipeline._crossencoder
    print("Models loaded successfully!\n")

    print("Type 'q' or 'quit' at any prompt to exit.\n")

    while True:
        try:
            doc_mode = input("Do you want to check a full document instead? (y/n): ").strip().lower()
            if doc_mode in ('q', 'quit'):
                break
                
            if doc_mode == 'y':
                print("\nType or paste your document (up to 10 sentences).")
                print("Press Enter on an empty line to execute.")
                lines = []
                while True:
                    line = input()
                    if line == '':
                        break
                    lines.append(line)
                
                doc_text = " ".join(lines).strip()
                if not doc_text:
                    continue
                    
                print("\n── Document Analysis ──────────────────────────────")
                res = pipeline.check_document(doc_text)
                stats = res['stats']
                sentences = nltk.sent_tokenize(doc_text)
                
                print(f"Total Sentences:    {len(sentences)}")
                print(f"Total Pairs:        {stats['total_pairs']}")
                print(f"Pairs After Filter: {stats['filtered_pairs']}")
                print(f"Filter Reduction:   {stats['filter_reduction_pct']}%\n")
                
                contradictions = res['contradictions']
                if not contradictions:
                    print("Final Verdict: Document is fully consistent.")
                else:
                    print(f"Contradictions Found: {len(contradictions)}")
                    print("\n── Contradictions List ──")
                    for idx, c in enumerate(contradictions, 1):
                        print(f"[{idx}] S1: {c['sentence1']}")
                        print(f"    S2: {c['sentence2']}")
                        print(f"    Confidence: {c['confidence']:.4f}\n")
                        
                print("───────────────────────────────────────────────\n")
                
            else:
                s1 = input("Sentence 1: ").strip()
                if s1.lower() in ('q', 'quit'):
                    break
                if not s1:
                    continue

                s2 = input("Sentence 2: ").strip()
                if s2.lower() in ('q', 'quit'):
                    break
                if not s2:
                    continue

                print("\n── Analysis ───────────────────────────────────")
                
                # SBERT Stage
                emb1 = biencoder.encode([s1])
                emb2 = biencoder.encode([s2])
                sim = float((emb1 * emb2).sum())
                
                passed_filter = sim >= SIMILARITY_THRESHOLD
                
                print(f"SBERT Cosine Similarity: {sim:.4f}")
                print(f"Passed SBERT Filter?     {'YES' if passed_filter else 'NO'} (Threshold: {SIMILARITY_THRESHOLD:.2f})")
                
                # CrossEncoder Stage
                if passed_filter:
                    probs = crossencoder.predict_pair(s1, s2)
                    
                    print("\nDeBERTa NLI Probabilities:")
                    print(f"  Entailment:    {probs.get('entailment', 0.0):.4f}")
                    print(f"  Neutral:       {probs.get('neutral', 0.0):.4f}")
                    print(f"  Contradiction: {probs.get('contradiction', 0.0):.4f}")
                    
                    # Verdict resolution
                    is_contradiction = probs.get('contradiction', 0.0) >= CONTRADICTION_THRESHOLD
                    verdict = "CONTRADICTION" if is_contradiction else "CONSISTENT"
                    confidence = probs.get('contradiction', 0.0) if is_contradiction else max(probs.get('entailment', 0.0), probs.get('neutral', 0.0))
                    
                    print(f"\nFinal Verdict: ** {verdict} **")
                    print(f"Confidence:    {confidence:.4f}")
                else:
                    # Bypassed
                    print("\nFinal Verdict: ** CONSISTENT **")
                    print("Reason:        Filtered — sentences are topically dissimilar, treated as consistent by default.")
                    print("Note:          To see full pipeline (SBERT + DeBERTa), use sentences about the same topic.")
                    print("Confidence:    N/A")
                    
                print("───────────────────────────────────────────────\n")
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
