import math
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import faiss
import boto3
import concurrent.futures as cf
from botocore.config import Config
from pandas.core.interchange.dataframe_protocol import DataFrame
from json_repair import repair_json

import logging

from shared_and_constants import *
from csv_net_log_classifier import classify_csv_file


boto3.set_stream_logger('boto3.resources', logging.INFO)

INCLUDE_NORMAL =False


EXCLUDE_FROM_NUMERIC = {
    "srcip", "dstip", "stime", "ltime", "attack_cat", "label", "id"
}
DEFAULT_CATEGORICAL = ["proto", "service", "state"]


#%%


#Helpers
def infer_numeric_columns(df: pd.DataFrame, class_col: str, cat_cols: List[str]) -> List[str]:

    """
    Infer candidate numeric feature columns for UNSW-NB15 profiling.

    This function examines the dataframe schema to select columns that are:
      - Numeric dtype per pandas type inference,
      - Not the attack class column,
      - Not in the declared categorical list,
      - Not in the exclusion set (identifiers/timestamps/labels).

    Args:
        df (pd.DataFrame): Input dataset (e.g., UNSW-NB15 formatted logs).
        class_col (str): Name of the class/label column (e.g. "attack_cat").
        cat_cols (List[str]): Columns to treat as categorical.

    Returns:
        List[str]: Names of columns to summarise as numeric features.

    Justification:
        - Keeps downstream summaries focused on meaningful metrics.
        - Excludes identifiers and categorical fields that would skew statistics.
        - Provides a deterministic, transparent feature selection baseline.
    """

    numeric_columns: List[str] = []
    categorical_set = set(cat_cols)

    for column_name in df.columns:
        if (
            column_name == class_col
            or column_name in categorical_set
            or column_name in EXCLUDE_FROM_NUMERIC
        ):
            continue

        if pd.api.types.is_numeric_dtype(df[column_name]):
            numeric_columns.append(column_name)

    return numeric_columns

def _to_float(input_value):
    """
    Safely convert a value to float with NaN/Inf protection.

    Converts inputs to float, returning 0.0 for None/NaN/Inf or conversion errors.
    This keeps statistical summaries robust for downstream JSON serialisation.

    Args:
        input_value (Any): Value to convert.

    Returns:
        float: Finite float value suitable for JSON.

    Justification:
        - Prevents serialisation and math errors in summaries.
        - Centralises defensive casting for consistency.
    """
    try:
        if input_value is None or (isinstance(input_value, float) and (math.isnan(input_value) or math.isinf(input_value))):
            return 0.0
        return float(input_value)
    except Exception:
        return 0.0

def safe_mode_and_freq(series: pd.Series):

    """
    Compute the mode and its relative frequency for a categorical series.

    Returns the most frequent (non-null) value and its fraction of the series.

    Args:
        series (pd.Series): A categorical-like column.

    Returns:
        Tuple[Any, float]: (mode_value, frequency in [0,1])

    Justification:
        - Provides compact categorical signal for profile→query text.
        - Helps LLM grounding by exposing dominant protocol/service/state.
    """

    if series.empty:
        return None, 0.0
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return None, 0.0
    mode_val = vc.index[0]
    freq = float(vc.iloc[0]) / float(len(series)) if len(series) else 0.0
    return mode_val, freq

def summarise_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Summarise numeric columns with mean/median/std.

    Args:
        df (pd.DataFrame): Dataset containing numeric columns.
        numeric_cols (List[str]): Columns to summarise.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of column→{"mean","median","std"}.

    Justification:
        - Supplies stable, interpretable statistics to class profiles.
        - Normalizes noise by focusing on robust summary metrics.
    """

    out = {}
    if df.empty or not numeric_cols:
        return out
    means   = df[numeric_cols].mean(numeric_only=True)
    medians = df[numeric_cols].median(numeric_only=True)
    stds    = df[numeric_cols].std(numeric_only=True, ddof=1)
    for col in numeric_cols:
        out[col] = {
            "mean":   _to_float(means.get(col)),
            "median": _to_float(medians.get(col)),
            "std":    _to_float(stds.get(col)),
        }
    return out

def add_global_means(overall_numeric: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

    """
    Add a 'global_mean' field to overall numeric summaries.

    Args:
        overall_numeric (Dict[str, Dict[str, float]]): Output of summarize_numeric.

    Returns:
        Dict[str, Dict[str, float]]: Same structure with 'global_mean' per column.

    Justification:
        - Establishes a baseline to compare per-class deviations.
        - Simplifies prompt construction and LLM reasoning about contrasts.
    """

    for key, val in overall_numeric.items():
        val["global_mean"] = _to_float(val.get("mean"))
    return overall_numeric

def attach_global_mean(per_class_numeric: Dict[str, Dict[str, float]], overall_numeric: Dict[str, Dict[str, float]]):
    """
    Attach the global mean onto each per-class numeric summary.

    Args:
        per_class_numeric (Dict[str, Dict[str, float]]): Stats for a single class.
        overall_numeric (Dict[str, Dict[str, float]]): Global stats for reference.

    Returns:
        None. (Mutates per_class_numeric in place.)

    Justification:
        - Allows the LLM to weigh whether a class deviates from the baseline.
        - Keeps schema compact and aligned across classes.
    """
    for k in per_class_numeric:
        g = overall_numeric.get(k, {}).get("global_mean")
        per_class_numeric[k]["global_mean"] = _to_float(g)

def summarise_categorical(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Produce categorical summaries (mode and frequency) for specified columns.

    Args:
        df (pd.DataFrame): Input dataset.
        cat_cols (List[str]): Columns to treat as categorical.

    Returns:
        Dict[str, Dict[str, Any]]: { col: {"mode": value, "freq": float} }

    Justification:
        - Encodes dominant protocol/service/state signals clearly.
        - Aids retrieval by turning distributions into clear descriptors.
    """

    out = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        mode_val, freq = safe_mode_and_freq(df[col].astype(str))
        out[col] = {"mode": mode_val, "freq": _to_float(freq)}
    return out

def build_class_profile(csv_path: Path, class_col="attack_cat", cat_cols=None, include_normal=False) -> Dict[str, Any]:
    """
    Build per-class numeric and categorical profiles from a CSV file.

    Pipeline:
        1) Load CSV, normalise class column.
        2) Infer numeric columns (excludes identifiers/categoricals).
        3) Summarise overall statistics and categorical modes.
        4) For each class (optionally excluding 'normal/benign'):
           - Compute numeric stats and attach global mean.
           - Compute categorical modes with frequencies.
        5) Add _analysis (overall stats) and _meta (schema) sections.

    Args:
        csv_path (Path): Path to CSV (UNSW-NB15-like).
        class_col (str): Label column; default "attack_cat".
        cat_cols (List[str] | None): Categorical columns (defaults to ["proto","service","state"]).
        include_normal (bool): Include 'normal/benign' in profiles if True.

    Returns:
        Dict[str, Any]: Multi-class profile with analysis and metadata.

    Justification:
        - Provides structured evidence for mapping classes → MITRE ATT&CK.
        - Keeps LLM grounded with statistical signals vs. raw rows.
    """
    df = pd.read_csv(csv_path)
    return build_class_profile_df(df)


def build_class_profile_df(df:DataFrame, class_col="attack_cat", cat_cols=None, include_normal=False) -> Dict[str, Any]:
    """
    Build per-class profiles from an in-memory DataFrame.

    Same as build_class_profile but uses an existing DataFrame (e.g. uploaded in the API).

    Args:
        df (pd.DataFrame): Dataset in memory.
        class_col (str): Label column.
        cat_cols (List[str] | None): Categorical columns.
        include_normal (bool): Include 'normal/benign' classes.

    Returns:
        Dict[str, Any]: Multi-class profile with analysis and metadata.

    Justification:
        - Admin testing of the structured mapping part of the application without user input

    """

    cat_cols = cat_cols or list(DEFAULT_CATEGORICAL)

    if class_col not in df.columns:
        raise ValueError(f"Class column '{class_col}' not found in CSV. Columns: {list(df.columns)}")

    df = classify_csv_file(df, STRUCTURED_PYCARET_ML_MODEL)

    df[class_col] = df[class_col].astype(str).str.strip()

    numeric_cols = infer_numeric_columns(df, class_col, cat_cols)
    overall_numeric = summarise_numeric(df, numeric_cols)
    overall_numeric = add_global_means(overall_numeric)
    overall_categorical = summarise_categorical(df, cat_cols)

    result = {"_analysis": {"numeric": overall_numeric,
                            "categorical": overall_categorical}}  ##********************************
    classes = sorted(df[class_col].dropna().unique().tolist())
    for cls in classes:
        if not include_normal and cls.lower() in {"normal", "benign"}:
            continue
        sub = df[df[class_col] == cls]
        cls_numeric = summarise_numeric(sub, numeric_cols)
        attach_global_mean(cls_numeric, overall_numeric)
        cls_categorical = summarise_categorical(sub, cat_cols)
        result[cls] = {"numeric": cls_numeric, "categorical": cls_categorical}

    result["_meta"] = {
        "class_column": class_col,
        "categorical_columns": cat_cols,
        "numeric_columns": numeric_cols,
        "n_rows": int(len(df))
    }
    return result

def as_query_text(profile: Dict[str, Any]) -> str:
    """
    Convert a multi-class profile to retrieval-friendly query text.

    The query includes, per non-normal class:
      - The class name,
      - Dominant categorical values (proto/service/state),
      - Selected numeric means for key traffic features (e.g. sload, sbytes).

    Args:
        profile (Dict[str, Any]): Output from build_class_profile(_df).

    Returns:
        str: A compact text description used for embedding and retrieval.

    Justification:
        - Bridges numeric/categorical summaries to semantic retrieval.
        - Boosts recall by hinting specific traffic signatures to the index.
    """
    lines = []
    for cls, payload in profile.items():
        if cls in {"_analysis", "_meta"}:
            continue
        if cls.lower() in {"normal","benign"}:
            continue
        lines.append(f"class: {cls}")
        cat = payload.get("categorical", {})
        num = payload.get("numeric", {})
        for k in ("proto","service","state"):
            mv = (cat.get(k) or {}).get("mode")
            if mv: lines.append(f"{k}: {mv}")
        for k in ("sload","rate","sbytes","spkts","dtcpb","stcpb","ct_dst_ltm","ct_src_ltm"):
            meanv = (num.get(k) or {}).get("mean")
            if meanv: lines.append(f"{k}_mean: {meanv}")
    return "\n".join(lines)


# BEDROCK (Titan)----------------------------------------------------------------------------------------------------------------------------------------------

def titan_embed_single(text: str, region: str, model_id: str) -> np.ndarray:

    """
    Embed a single string via Amazon Titan on Bedrock with normalisation.

    Handles both canonical and alternative payload shapes that providers may return.

    Args:
        text (str): Input text to embed.
        region (str): AWS region hosting Bedrock.
        model_id (str): Titan model identifier.

    Returns:
        np.ndarray: 1D normalised embedding vector.

    Raises:
        RuntimeError: If the Titan response has unexpected shape.

    Justification:
        - Provides robust, provider-agnostic embedding extraction.
        - Normalisation aligns with FAISS inner-product or cosine usage.
    """


    rt = boto3.client("bedrock-runtime")

    body = {"inputText": text}  # Titan expects a *string*, not an array
    resp = rt.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    # Titan returns {"embedding": [...]}
    vec = payload.get("embedding")
    if vec is None:
        # Rare alt shapes from some SDKs/providers
        if "embeddings" in payload and payload["embeddings"]:
            vec = payload["embeddings"][0].get("embedding")
    if vec is None:
        raise RuntimeError(f"Unexpected Titan response: {payload.keys()}")
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v

def titan_embed_batch(texts: list, region: str, model_id: str, workers: int = 2) -> np.ndarray:
    """
    Embed multiple strings concurrently using Titan single-text calls.

    Titan expects single-text `inputText`; this function parallelises many
    single calls with a thread pool to achieve batch-like throughput.

    Args:
        texts (list): Input strings to embed.
        region (str): Bedrock region.
        model_id (str): Titan model ID.
        workers (int): Thread pool size; tune for CPU/network limits.

    Returns:
        np.ndarray: [N, D] normalised embedding matrix.

    Justification:
        - Increases throughput while respecting Titan request schema.
        - Keeps memory bounded compared to large monolithic batches.
    """

    def _one(t: str) -> np.ndarray:
        return titan_embed_single(t, region=region, model_id=model_id)

    vecs = []
    # Use a thread pool to parallelize single-text calls
    chuck_counter = 0
    total_chunks = len(texts)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for v in ex.map(_one, texts):
            chuck_counter += 1
            print(f'Proccessing: {chuck_counter}/{len(texts)}')
            vecs.append(v)
    # Ensure a 2D array
    M = np.vstack(vecs).astype(np.float32)
    # Row-normalize for cosine/IP
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n

# RETRIEVAL -----------------------------------------------------------------------------------------------------------------------------------------------

def retrieve(index_dir: Path, qvec: np.ndarray, topk: int = 8) -> List[Dict[str, Any]]:

    """
    Retrieve the top-k knowledge documents for a given query vector.

    Uses FAISS inner-product search if available; otherwise computes
    similarities against a dense NumPy matrix and ranks manually.

    Args:
        index_dir (Path): Directory holding index artefacts.
        qvec (np.ndarray): 1D normalised query embedding.
        topk (int): Number of results to return.

    Returns:
        List[Dict[str, Any]]: Results with score, id, name, type, snippet, payload.

    Justification:
        - Abstracts over FAISS vs. dense fallback transparently.
        - Returns human-inspectable snippets to aid debugging/grounding.
    """

    meta = json.loads((index_dir / "artifact.json").read_text(encoding="utf-8"))
    docs = [json.loads(l) for l in (index_dir / "docs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    id_map = json.loads((index_dir / "id_map.json").read_text(encoding="utf-8"))

    results = []
    if meta.get("use_faiss"):

        index = faiss.read_index(str(index_dir / "index.faiss"))
        D, I = index.search(qvec.reshape(1,-1).astype(np.float32), topk)
        scores, idx = D[0].tolist(), I[0].tolist()
    else:
        X = np.load(index_dir / "dense.npy")
        sims = X @ qvec.astype(np.float32)
        idx = np.argsort(-sims)[:topk].tolist()
        scores = sims[idx].tolist()

    for i, sc in zip(idx, scores):
        meta = id_map[str(i)]
        payload = docs[int(i)]["payload"]
        text = docs[int(i)]["text"] if "text" in docs[int(i)] else ""
        snippet = (text or "")[:400].replace("\n"," ")
        results.append({
            "score": float(sc),
            "id": meta["id"],
            "name": meta["name"],
            "type": meta["type"],
            "snippet": snippet,
            "payload": payload
        })
    return results

# PROMPT + NOVA MICRO --------------------------------------------------------------------------------------------------------------------------------


def build_llm_messages(class_profile, retrieved):

    """
    Construct a Nova Micro prompt to map UNSW-NB15 classes to ATT&CK.

    The prompt includes:
        - A system message defining the assistant’s role.
        - A JSON schema describing the expected output.
        - The class profile as JSON (grounding).
        - The retrieved context (top-k) as JSON (evidence).

    Args:
        class_profile (Dict[str, Any]): Output from build_class_profile(_df).
        retrieved (List[Dict[str, Any]]): Top-k retrieval results.

    Returns:
        Dict[str, Any]: A payload with 'system' and 'messages' for Bedrock.

    Justification:
        - Enforces structured output via schema.
        - Grounds the model in real data to reduce hallucinations.
    """

    column_info_text = None


    try:
        col_meta_df = pd.read_csv(STRUCTURED_UNSW_COL_DEFS)
        # Create concise column → description mapping
        column_info_text = "\n".join(
            f"- {row['Name']}: {row['Description']}" for _, row in col_meta_df.iterrows()
            if 'Name' in row and 'Description' in row
        )
    except Exception as e:
        column_info_text = "Column metadata unavailable."


    system_text = (
        "You are a cybersecurity assistant that maps UNSW-NB15 class profiles "
        "to MITRE ATT&CK tactics, techniques, and sub-techniques.\n\n"
        "Use the provided UNSW-NB15 column descriptions to interpret the statistical "
        "signals within class profiles accurately. Each column represents a specific "
        "feature in a network flow, such as packet counts, bytes transferred, duration, "
        "and protocol-level flags.\n\n"
        "UNSW-NB15 Column Descriptions:\n"
        f"{column_info_text}\n\n"
        "Return ONLY valid JSON that matches the schema, no extra text."
    )

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "attack_class": {"type": "string"},
                "tactics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tactic_id": {"type": "string"},
                            "tactic_name": {"type": "string"},
                            "justification": {"type": "string"},
                            "confidence": {"type": "number"},
                            "techniques": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "confidence": {"type": "number"}
                                    },
                                    "required": ["id", "name", "confidence"]
                                }
                            }
                        },
                        "required": ["tactic_id", "tactic_name", "justification", "confidence", "techniques"]
                    }
                }
            },
            "required": ["attack_class", "tactics"]
        }
    }

    instructions = (
        "Produce a JSON array per this JSON Schema (below). "
        "For each attack class (exclude 'Normal'), output:\n"
        "1. Likely ATT&CK tactics with justification and confidence (0-1).\n"
        "2. ~3 most probable techniques or sub-techniques (id, name, confidence).\n\n"
        "Use the UNSW-NB15 feature definitions above to reason about what each "
        "numeric or categorical signal implies (e.g., flow duration, packet count, "
        "TCP flags, bytes exchanged) when mapping to ATT&CK behaviors. "
        "Base your reasoning ONLY on the retrieved context and class profile. "
        "If evidence is weak, reduce confidence rather than inventing content."
    )






    return {
        # IMPORTANT: system must be an ARRAY of content blocks
        "system": [{"text": system_text}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": instructions},
                    {"text": "JSON Schema:\n" + json.dumps(schema, indent=2)},
                    {"text": "Class Profile:\n```json\n" + json.dumps(class_profile, indent=2) + "\n```"},
                    {"text": "Retrieved Context (top-k):\n```json\n" + json.dumps(retrieved, indent=2) + "\n```"},
                    {"text": "Return ONLY the JSON array (no markdown, no pre/post text)."}
                ]
            }
        ]
    }


def call_nova_micro(msgs, region: str, model_id: str, max_tokens: int = 2048):

    """
    Invoke Nova Micro and parse/repair the JSON response.

    Uses Bedrock Runtime with adaptive retries. Tries multiple common fields
    for the provider’s output and applies json_repair to salvage slightly
    malformed JSON from the LLM.

    Args:
        msgs (Dict[str, Any]): Prompt constructed by build_llm_messages.
        region (str): Bedrock region (typically 'us-east-1').
        model_id (str): Nova Micro model ID.
        max_tokens (int): Token cap for the response.

    Returns:
        Any: Parsed Python object from the model's JSON output.

    Justification:
        - Handles real-world LLM outputs that may include code fences or minor issues.

    """

    rt = boto3.client("bedrock-runtime", region_name='us-east-1',
                      config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))

    body = {
        "system": msgs["system"],                 # <-- array of {"text": "..."}
        "messages": msgs["messages"],             # <-- list of {role, content:[{text:...}]}
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": 0.2,
            "topP": 0.9
        }
    }

    resp = rt.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())

    # Typical Converse-style response shape:
    # {"output":{"message":{"role":"assistant","content":[{"text":"..."}]}}}
    txt = None
    try:
        txt = payload["output"]["message"]["content"][0]["text"]
    except Exception:
        # fallback if the provider wrapper differs slightly
        for key in ("outputText","generated_text","text"):
            if key in payload:
                txt = payload[key]; break
        if txt is None:
            txt = json.dumps(payload)

    cleaned_text = clean_nova_response(txt)

    # Expect JSON array from the model
    try:
        return json.loads( repair_json(cleaned_text))                                                                            ##*******************************
    except Exception as e:
        logging.exception(f"Failed to parse Nova Micro response: {cleaned_text}")
        raise e