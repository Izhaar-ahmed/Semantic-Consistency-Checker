"""Evaluate the consistency pipeline on the STS-B test set with baselines."""

import json
import time

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from data import load_stsb
from pipeline import ConsistencyPipeline
from biencoder import BiEncoder
from crossencoder import NLICrossEncoder
from config import SIMILARITY_THRESHOLD

def get_ce_score(scores):
    return scores["entailment"] * 5.0 + scores["neutral"] * 2.5 + scores["contradiction"] * 0.0

def evaluate_on_stsb() -> dict:
    print("[1/3] Loading STS-B test split …")
    stsb = load_stsb()
    test_df = stsb["test"]
    
    n_pairs = len(test_df)
    s1_list = test_df["sentence1"].tolist()
    s2_list = test_df["sentence2"].tolist()
    gold_scores = (test_df["score"]).tolist()
    gold_labels = test_df["label"].tolist()
    binary_gold = [1 if g == "contradiction" else 0 for g in gold_labels]

    print("[2/3] Evaluating Baselines and Pipeline …")
    
    # Instantiate components
    biencoder = BiEncoder()
    crossencoder = NLICrossEncoder()
    
    # ---------------------------------------------------------
    # Baseline 1: SBERT only
    # ---------------------------------------------------------
    print("  Running Baseline 1 (SBERT only)...")
    start = time.time()
    # Batch encode
    emb1 = biencoder.encode(s1_list)
    emb2 = biencoder.encode(s2_list)
    sims = (emb1 * emb2).sum(axis=1) # cosine similarity for aligned pairs
    pred_scores_sbert = [float(s) * 5.0 for s in sims]
    
    binary_pred_sbert = [1 if (s < 2.0) else 0 for s in pred_scores_sbert]
    time_sbert = time.time() - start

    # ---------------------------------------------------------
    # Baseline 2: CrossEncoder only
    # ---------------------------------------------------------
    print("  Running Baseline 2 (CrossEncoder only)...")
    start = time.time()
    pairs = list(zip(s1_list, s2_list))
    ce_results = crossencoder.predict_batch(pairs)
    
    pred_scores_ce = [get_ce_score(res) for res in ce_results]
    # Prediction of contradiction: using threshold or just the highest score?
    # Original evaluate.py used labels
    binary_pred_ce = [1 if res["contradiction"] >= 0.70 else 0 for res in ce_results]
    time_ce = time.time() - start

    # ---------------------------------------------------------
    # Our Pipeline: SBERT filter + CrossEncoder
    # ---------------------------------------------------------
    print("  Running Pipeline (SBERT filter + CrossEncoder)...")
    start = time.time()
    pred_scores_pipe = []
    binary_pred_pipe = []
    
    # we already have sims from SBERT. In real life we filter those >= threshold.
    # We will only run CE on pairs where sim >= SIMILARITY_THRESHOLD.
    # To be fair to the pipeline timing, we should measure encode+filter+CE.
    pipe_start = time.time()
    e1 = biencoder.encode(s1_list)
    e2 = biencoder.encode(s2_list)
    pipe_sims = (e1 * e2).sum(axis=1)
    
    filtered_pairs_indices = [i for i, s in enumerate(pipe_sims) if s >= SIMILARITY_THRESHOLD]
    filtered_pairs = [(s1_list[i], s2_list[i]) for i in filtered_pairs_indices]
    
    if filtered_pairs:
        pipe_ce_results = crossencoder.predict_batch(filtered_pairs)
    else:
        pipe_ce_results = []
        
    ce_idx = 0
    pipe_examples = [] # to store rich results for the prompt
    for i in range(n_pairs):
        s1 = s1_list[i]
        s2 = s2_list[i]
        sim = pipe_sims[i]
        if sim >= SIMILARITY_THRESHOLD:
            ce_res = pipe_ce_results[ce_idx]
            ce_idx += 1
            score = get_ce_score(ce_res)
            is_contra = 1 if ce_res["contradiction"] >= 0.70 else 0
            entail_conf = ce_res["entailment"]
            contra_conf = ce_res["contradiction"]
        else:
            score = float(sim) * 5.0
            # if filtered by SBERT and sim < threshold, is it contradiction? 
            # Yes, if we assume low similarity = contradiction or neutral. 
            is_contra = 1 if score <= 2.0 else 0
            entail_conf = 0.0
            contra_conf = 1.0 if is_contra else 0.0
            
        pred_scores_pipe.append(score)
        binary_pred_pipe.append(is_contra)
        
        pipe_examples.append({
            "s1": s1, "s2": s2, "gold": gold_scores[i], 
            "is_contra": is_contra, "entail_conf": entail_conf, "contra_conf": contra_conf
        })
        
    time_pipe = time.time() - pipe_start
    
    # Metrics calculation
    def calc_metrics(preds, binary_preds):
        p, _ = pearsonr(preds, gold_scores)
        s, _ = spearmanr(preds, gold_scores)
        f1 = f1_score(binary_gold, binary_preds, zero_division=0)
        return p, s, f1

    p_sbert, s_sbert, f1_sbert = calc_metrics(pred_scores_sbert, binary_pred_sbert)
    p_ce, s_ce, f1_ce = calc_metrics(pred_scores_ce, binary_pred_ce)
    p_pipe, s_pipe, f1_pipe = calc_metrics(pred_scores_pipe, binary_pred_pipe)

    print("\n[3/3] ── Results ──────────────────────────")
    print(f"{'Method':<20} | {'Pearson r':<10} | {'Spearman r':<10} | {'F1 Contra':<10} | {'Time (s)':<10}")
    print("-" * 70)
    print(f"{'Baseline 1 (SBERT)':<20} | {p_sbert:<10.4f} | {s_sbert:<10.4f} | {f1_sbert:<10.4f} | {time_sbert:<10.2f}")
    print(f"{'Baseline 2 (CE)':<20} | {p_ce:<10.4f} | {s_ce:<10.4f} | {f1_ce:<10.4f} | {time_ce:<10.2f}")
    print(f"{'Our Pipeline':<20} | {p_pipe:<10.4f} | {s_pipe:<10.4f} | {f1_pipe:<10.4f} | {time_pipe:<10.2f}")

    # Extract 5 correctly predicted contradictions and 5 consistent
    print("\n── 5 Correctly Predicted Contradictions ──")
    contra_examples = [e for e in pipe_examples if (e["gold"] * 5.0) < 1.5 and e["contra_conf"] > 0.7][:5]
    for i, e in enumerate(contra_examples, 1):
        print(f"{i}. S1: {e['s1']}")
        print(f"   S2: {e['s2']}")
        print(f"   Human Score: {e['gold']*5.0:.2f} | Confidence: {e['contra_conf']:.4f}")

    print("\n── 5 Correctly Predicted Consistent ──")
    consist_examples = [e for e in pipe_examples if (e["gold"] * 5.0) > 4.0 and e["entail_conf"] > 0.8][:5]
    for i, e in enumerate(consist_examples, 1):
        print(f"{i}. S1: {e['s1']}")
        print(f"   S2: {e['s2']}")
        print(f"   Human Score: {e['gold']*5.0:.2f} | Confidence: {e['entail_conf']:.4f}")
        
    return {}

if __name__ == "__main__":
    evaluate_on_stsb()
