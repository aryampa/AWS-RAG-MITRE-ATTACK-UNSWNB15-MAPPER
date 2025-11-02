import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np

from structured_helpers import  titan_embed_batch
from shared_and_constants import *


def load_docs_jsonl(p: Path) -> List[Dict[str, Any]]:
    """
    Load JSONL-formatted documents from the disk.

    Args:
        p (Path): Path to a .jsonl file (one JSON object per line).

    Returns:
        List[Dict[str, Any]]: Parsed documents.

    """
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_doc_text(rec: Dict[str, Any]) -> str:
    """
    Flatten a MITRE/knowledge record into a retrieval text blob.

    Includes:
        - Basic header: [type] id name
        - Description and detection notes (if present)
        - UNSW mappings (class name, natural-language profile, statistical signals,
          disambiguation tips, and tactic hints)

    Args:
        rec (Dict[str, Any]): A knowledge record (or its payload).

    Returns:
        str: Plain text suitable for Titan embeddings.

    Justification:
        - Concatenates salient fields to maximise retrieval quality.
        - Retains alignment between UNSW signals and ATT&CK objects.
    """

    parts = [
        f"[{rec.get('type', '')}] {rec.get('id', '')} {rec.get('name', '')}",
        rec.get("description", "") or "",
        rec.get("detection", "") or "",
    ]
    for m in rec.get("unsw_mappings", []) or []:
        parts += [
            f"attack_class: {m.get('attack_class', '')}",
            "natural_language_profile: " + " | ".join(m.get("natural_language_profile", [])),
            "statistical_signals: " + " | ".join(m.get("statistical_signals", [])),
            "disambiguation_tips: " + " | ".join(m.get("disambiguation_tips", [])),
            f"tactic_hint: {m.get('tactic', '')}"
        ]
    return "\n".join([p for p in parts if p]).strip()


def ensure_index(docs_jsonl: Path, index_dir: Path, region: str, embed_model_id: str):
    """
    Ensure a dense vector index exists for the knowledge base.

    Behavior:
        - If artefacts already exist, reuse them.
        - Otherwise:
            1) Load JSONL docs and flatten to text via build_doc_text.
            2) Embed with Titan (batch, normalised).
            3) Try FAISS IndexFlatIP, else save dense numpy fallback.
            4) Persist id_map, docs, and artefact metadata.

    Args:
        docs_jsonl (Path): Source knowledge JSONL file.
        index_dir (Path): Target directory for index artifacts.
        region (str): Bedrock region for embeddings.
        embed_model_id (str): Titan embed model ID.

    Returns:
        None (writes artifacts on disk).

    Justification:
        - One-time cost to speed up subsequent retrievals.
        - Supports FAISS for fast IP search, with a graceful fallback.
    """

    index_dir.mkdir(parents=True, exist_ok=True)
    # If index already exists, return
    if (index_dir / "artifact.json").exists() and (
            (index_dir / "index.faiss").exists() or (index_dir / "dense.npy").exists()):
        info(f"Using existing index at {index_dir}")
        return

    info("Building Titan embeddings index...")
    docs = load_docs_jsonl(docs_jsonl)
    texts = [build_doc_text(d["payload"]) for d in docs] if "payload" in docs[0] else [build_doc_text(d) for d in docs]

    vecs = titan_embed_batch(texts, region=region, model_id=embed_model_id)
    vecs = l2_normalize(vecs)

    use_faiss = False
    try:

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, str(index_dir / "index.faiss"))
        use_faiss = True
        info(f"Wrote FAISS index: {index_dir / 'index.faiss'}")
    except Exception as e:
        warn(f"FAISS not available or failed ({e}); using dense numpy fallback")
        np.save(index_dir / "dense.npy", vecs)

    id_map = {i: {"id": (docs[i].get("id") if "id" in docs[i] else docs[i]["payload"].get("id")),
                  "name": (docs[i].get("name") if "name" in docs[i] else docs[i]["payload"].get("name")),
                  "type": (docs[i].get("type") if "type" in docs[i] else docs[i]["payload"].get("type"))}
              for i in range(len(docs))}
    (index_dir / "id_map.json").write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    (index_dir / "docs.jsonl").write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    meta = {"use_faiss": use_faiss, "model_id": embed_model_id, "region": region, "num_docs": len(docs)}
    (index_dir / "artifact.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    info("Index build complete.")


def l2_normalize(M: np.ndarray) -> np.ndarray:
    """
    Row-normalise a 2D matrix for cosine/inner-product similarity.

    Args:
        M (np.ndarray): [N, D] embedding matrix.

    Returns:
        np.ndarray: Row-normalised matrix, safe for zero norms.

    Justification:
        - Ensures comparable magnitudes across vectors for scoring.
        - Avoids division-by-zero by clamping zero norms to 1.0.
    """

    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n
