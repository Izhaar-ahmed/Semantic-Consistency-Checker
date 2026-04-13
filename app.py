"""Flask API backend for the Semantic Consistency Checker UI."""

import json
import time
import nltk
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from pipeline import ConsistencyPipeline
from config import SIMILARITY_THRESHOLD, CONTRADICTION_THRESHOLD

nltk.download("punkt_tab", quiet=True)

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# Load pipeline once at startup
print("Loading pipeline models... (this may take a moment)")
pipeline = ConsistencyPipeline()
biencoder = pipeline._biencoder
crossencoder = pipeline._crossencoder
print("Models loaded successfully!")


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/check-pair", methods=["POST"])
def check_pair():
    """Analyze a single sentence pair."""
    data = request.get_json()
    s1 = data.get("sentence1", "").strip()
    s2 = data.get("sentence2", "").strip()

    if not s1 or not s2:
        return jsonify({"error": "Both sentences are required."}), 400

    start = time.time()

    # Stage 1: SBERT
    emb1 = biencoder.encode([s1])
    emb2 = biencoder.encode([s2])
    similarity = float((emb1 * emb2).sum())
    passed_filter = similarity >= SIMILARITY_THRESHOLD

    result = {
        "similarity": round(similarity, 4),
        "threshold": SIMILARITY_THRESHOLD,
        "passed_filter": passed_filter,
    }

    # Stage 2: CrossEncoder (only if passed)
    if passed_filter:
        probs = crossencoder.predict_pair(s1, s2)
        is_contradiction = probs.get("contradiction", 0.0) >= CONTRADICTION_THRESHOLD
        verdict = "CONTRADICTION" if is_contradiction else "CONSISTENT"
        confidence = probs.get("contradiction", 0.0) if is_contradiction else max(
            probs.get("entailment", 0.0), probs.get("neutral", 0.0)
        )
        result.update({
            "entailment": round(probs.get("entailment", 0.0), 4),
            "neutral": round(probs.get("neutral", 0.0), 4),
            "contradiction": round(probs.get("contradiction", 0.0), 4),
            "verdict": verdict,
            "confidence": round(confidence, 4),
        })
    else:
        result.update({
            "entailment": None,
            "neutral": None,
            "contradiction": None,
            "verdict": "CONSISTENT",
            "confidence": None,
            "filtered_reason": "Sentences are topically dissimilar — treated as consistent by default.",
        })

    result["time_ms"] = round((time.time() - start) * 1000, 1)
    return jsonify(result)


@app.route("/api/check-document", methods=["POST"])
def check_document():
    """Analyze a full document for contradictions."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Document text is required."}), 400

    start = time.time()
    res = pipeline.check_document(text)
    elapsed = time.time() - start

    sentences = nltk.sent_tokenize(text)

    return jsonify({
        "total_sentences": len(sentences),
        "total_pairs": res["stats"]["total_pairs"],
        "filtered_pairs": res["stats"]["filtered_pairs"],
        "filter_reduction_pct": res["stats"]["filter_reduction_pct"],
        "consistency_score": res["consistency_score"],
        "contradictions": res["contradictions"],
        "time_ms": round(elapsed * 1000, 1),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
