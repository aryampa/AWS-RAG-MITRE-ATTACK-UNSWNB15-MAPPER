
# - POST /analyser: accepts PDF, extracts text (pdfminer.six low-level API), retrieves context from FAISS,
#   calls Nova Micro LLM, returns JSON

import io
import logging
import re

from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
import boto3
from botocore.config import Config
from flask import jsonify
from json_repair import repair_json
from numpy.f2py.auxfuncs import throw_error

# ---- pdfminer.six low-level imports (no pdfminer.high_level) ----
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter



from shared_and_constants import *



# -------------------- Config --------------------

bedrock_region: str = BEDROCK_REGION
titan_model_id: str = BEDROCK_TITAN_MODEL_ID
nova_model_id: str = BEDROCK_LLM_MODEL_ID

index_dir: Path = UNSTRUCTURED_INDEX_DIR
max_chunks: int = UNSTRUCTURED_MAX_CHUNKS     # limit #chunks extracted from PDF
chunk_size: int = UNSTRUCTURED_CHUNK_SIZE    # characters per chunk
chunk_overlap: int = UNSTRUCTURED_CHUNK_OVERLAP
top_k_per_chunk: int = UNSTRUCTURED__TOP_K_PER_CHUNK
top_contexts: int = UNSTRUCTURED_TOP_CONTEXTS

# -------------------- Bedrock clients --------------------
def bedrock_client():
    """
    Create and return a boto3 Bedrock Runtime client with retry policy.
    The runtime client is used to invoke Titan embedding model.
    The region specifies the AWS region where the model is deployed.
    The config max_attempts specifies the maximum number of retries in case of a service error, before giving up.
    The mode is set to "standard" to allow retries on throttling and throttling errors.
    The 'bedrock-runtime' argument specifies the client to use.
    """
    return boto3.client(
        'bedrock-runtime',
        region_name=bedrock_region,
        config=Config(retries={'max_attempts': 10, 'mode': 'standard'})
    )

# -------------------- Embeddings --------------------
def titan_embed_texts(texts: List[str]) -> np.ndarray:

    """
    Generate Titan embeddings for a list of input texts (chunks of text).

    This function sequentially embeds each input string using the Amazon Titan
    embedding model hosted on Bedrock. Each embedding vector is then L2-normalised
    for cosine similarity. Handles minor schema variations in model responses.

    Cosine similarity explained: https://www.youtube.com/watch?v=zcUGLp5vwaQ

    Args:
        texts (List[str]): List of strings to embed.

    Returns:
        np.ndarray: A 2D NumPy array of shape [N, D], where:
            - N = number of input texts (chunks)
            - D = embedding dimension (model-dependent, Titan Text v2 = 1024 dimensions by default)

    Justification:
        - Normalization ensures vectors are comparable for similarity search.
        - Wrapping Bedrock calls here keeps retry/error handling consistent.
        - Provides safe parsing of different Titan output formats.
    """



    client = bedrock_client() # initialise bedrock client

    vectors = [] #will hold our array of vectors (number_of_chunks X embedding_dimensions)

    for text in texts: #for each chunk of text...
        body = {"inputText": text}

        # send the chunk out to return it's embedding (1 X 1024 vector)
        resp = client.invoke_model(
            modelId=titan_model_id,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )

        #extract the embedding from the response
        payload = json.loads(resp["body"].read())

        """
        Assume response format:   { "embedding": []}
        """

        chunk_embedding_vector = payload.get("embedding")
        if chunk_embedding_vector is None and "embeddings" in payload and payload["embeddings"]:

            """
            Next, assume response format:   {"embeddings": [ { "embedding": []}]},
            
            throwing an error if neither formats found in response.
            """

            chunk_embedding_vector = payload["embeddings"][0].get("embedding")
        if chunk_embedding_vector is None:
            raise RuntimeError(f"Unexpected Titan response keys: {list(payload.keys())}")

        #convert embedding to a numpy array
        v = np.asarray(chunk_embedding_vector, dtype=np.float32)

        n = float(np.linalg.norm(v)) # compute L2 norm (||v||2) as a scalar

        """
        Guarding against division by zero:
        -If np.linalg.norm(v) returns a 0-D ndarray; float(...) turns it into a Python scalar for the comparison (example: float(np.linalg.norm(np.array([0.0, 0.0], dtype=np.float32)) == 0.0)
        -The guard avoids dividing by 0 if an all-zeros vector ever appears. 
        -If n == 0, you simply skip normalisation; the zero vector is still appended (it will have no similarity with anything).
        """
        if n > 0:
            v = v / n                # rescale to unit length (||v'||2 = 1)

        vectors.append(v)

        #vstack turns our list of 1-D arrays into a 2-D array
    return np.vstack(vectors).astype(np.float32)

def titan_embed_single(text: str) -> np.ndarray:
    """
    Embed a single string using Titan.

    Shortcut function that wraps `titan_embed_texts` for convenience,
    ensuring consistent normalisation and error handling.

    Args:
        text (str): Input string.

    Returns:
        np.ndarray: A 1-D embedding vector.

    Justification:
        - Avoids duplicate logic in callers needing single embeddings.
        - Keeps interface consistent between batch and single operations.
    """
    return titan_embed_texts([text])[0]

# -------------------- Index loading --------------------
def load_index() -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:

    """
    Load FAISS index and associated artefacts from the disk.

    Loads three artefacts from the unstructured index directory:
        - "index.faiss" (vector index)
        - "docs.jsonl" (source documents for retrieval)
        - "id_map.json" (maps FAISS IDs to documents)

    Returns:
        Tuple containing:
            - faiss.Index: Trained FAISS index for similarity search.
            - List[Dict[str, Any]]: Source documents as JSON objects.
            - Dict[str, Any]: Mapping of FAISS index IDs to document metadata.

    Raises:
        FileNotFoundError: If required artefacts are missing.

    Justification:
        - Encapsulates index loading in one place for reusability.
        - Ensures all retrieval calls operate on consistent artefacts.
        - Centralises error handling for missing/corrupt index files.
    """


    index_path = index_dir / "index.faiss"
    docs_path = index_dir / "docs.jsonl"
    id_map_path = index_dir / "id_map.json"

    if not (index_path.exists() and docs_path.exists() and id_map_path.exists()):
        raise FileNotFoundError(f"Index artifacts not found under {index_dir}")

    index = faiss.read_index(str(index_path))
    docs = [json.loads(l) for l in docs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    return index, docs, id_map

faiss_index, source_docs, id_map = load_index() # load index (Todo: explore cacheing load_index result)

# -------------------- PDF text extraction (pdfminer.six low-level) --------------------

def preserve_paragraphs(text:str) -> str:
    # Light-structure clean-up (keeps paragraph breaks)
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse big gaps
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()
    return text


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF using pdfminer low-level API.

    Reads raw PDF bytes, iterates over pages, and extracts visible text while
    normalising whitespace. Avoids high-level pdfminer APIs for fine-grained control.

    Args:
        pdf_bytes (bytes): Raw binary content of a PDF.

    Returns:
        str: Extracted plain text.

    Justification:
        - Provides maximum flexibility over text extraction (e.g., layout tweaks).
        - Ensures text is normalised for downstream chunking and retrieval.
    """

    output_buffer = io.StringIO()
    laparams = LAParams()  #  tweak for finer control (TOdo: explore other tweaks, if needed)

    """
    ResourceManager facilitates reuse of shared resources such as fonts and images so that large objects are not allocated multiple times.
    
    """
    rsrcmgr = PDFResourceManager()

    """
    
    <var>interpreter</var> facilitates the process of converting PDF pages into text.
    <var>device</var> facilitates the process of converting the extracted layers into text.
    <var>output_buffer</var>,  a StringIO object,  stores the extracted text

    """
    with io.BytesIO(pdf_bytes) as fp:
        with TextConverter(rsrcmgr, output_buffer, laparams=laparams) as device:


            interpreter = PDFPageInterpreter(rsrcmgr, device)

            #all this  loop does  is:  read page → convert to text → append to output_buffer
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
    text = output_buffer.getvalue() # get accumulated text
    output_buffer.close() #close StringIO stream

    # basic clean-up
    #collapses whitespace, new-line and tabs into a single space, and all text to single line
    text = re.sub(r"\s+", " ", text).strip()

    #preserve paragraphs
    #Todo: test later
    #text = preserve_paragraphs(text)



    return text



def chunk_text(text: str, size: int, overlap: int, limit: int) -> list[str]:
    """
    Split text into overlapping windows; stop at the tail without duplicating it.

    Args:
        text (str): Input text to chunk.
        size (int): Maximum number of characters per chunk.
        overlap (int): Number of overlapping characters between chunks.
        limit (int): Maximum number of chunks.

    Returns:
        List[str]: List of text chunks.

    Justification:
        - Overlap preserves context across chunk boundaries.
        - Limit prevents extreme memory usage for long documents, especially if overlap is large. However, there is risk involving loss of information.
        - Limit also helps with cost management, and latency, especially for large documents.
        - Chunks improve retrieval quality by providing finer granularity.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0 (got {size})")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, size-1] (got {overlap}, size={size})")

    chunks = []
    input_text_length = len(text)
    start = 0
    step = size - overlap

    while start < input_text_length and len(chunks) < limit:
        end = min(input_text_length, start + size)
        chunks.append(text[start:end])
        if end == input_text_length:            # reached the tail once—stop
            break
        start += step                           # clean forward step; no stall

    return chunks


# -------------------- Retrieval --------------------
def retrieve_for_chunks(chunks: List[str]) -> List[Dict[str, Any]]:

    """
    Retrieve top MITRE ATT&CK documents for each chunk.

    For each text chunk:
        1. Embed with Titan embeddings.
        2. Perform FAISS search against the ATT&CK knowledge index.
        3. Aggregate scores per document (keep best per doc).
        4. Return the top-ranked contexts.

    Args:
        chunks (List[str]): Text chunks.

    Returns:
        List[Dict[str, Any]]: Retrieved docs with fields:
            - score (float)
            - id (str)
            - name (str)
            - type (str)
            - snippet (str)
            - payload (dict)

    Justification:
        - Chunk-level search increases recall of relevant context.
        - Aggregation ensures documents with multiple hits are ranked properly.
        - Provides both metadata and snippets for interpretability.
    """

    query_vecs = titan_embed_texts(chunks)   # [chunk_count, D], normalised
    scores = []
    idxs = []
    D, I = faiss_index.search(query_vecs, top_k_per_chunk)
    for row_d, row_i in zip(D, I):
        scores.extend(row_d.tolist())
        idxs.extend(row_i.tolist())

    agg: Dict[int, float] = {}
    for sc, i in zip(scores, idxs):
        if i < 0:
            continue
        agg[i] = max(agg.get(i, -1e9), sc)  # keep best per doc

    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_contexts]
    results = []
    for i, sc in top:
        doc = source_docs[i]
        payload = doc if "payload" not in doc else doc["payload"]
        header = f"[{payload.get('type','')}] {payload.get('id','')} {payload.get('name','')}".strip()
        desc = (payload.get("description") or "")[:400]
        snippet = (header + " | " + desc).strip()
        results.append({
            "score": float(sc),
            "id": payload.get("id"),
            "name": payload.get("name"),
            "type": payload.get("type"),
            "snippet": snippet,
            "payload": payload
        })
    return results

# -------------------- Nova Micro call --------------------
def build_nova_prompt(retrieved: List[Dict[str, Any]], report_summary: str) -> Dict[str, Any]:
    """
    Build a prompt for Nova Micro to map reports to MITRE ATT&CK.

    Combines:
        - System role: instructs model to behave as cybersecurity analyst.
        - Report summary: first ~1500 characters of extracted report text.
        - Retrieved context: top-k ATT&CK docs from FAISS.

    Args:
        retrieved (List[Dict[str, Any]]): Context documents.
        report_summary (str): Condensed report text.

    Returns:
        Dict[str, Any]: Payload formatted for Bedrock Nova Micro API.

    Justification:
        - Retrieval-augmented prompting constrains LLM output to valid ATT&CK mappings.
        - Including both summary + context improves grounding and reduces hallucination.
        - JSON-only instruction enforces structured output.
    """
    schema: Dict[str, Any] = {
  "type": "object",
  "properties": {
    "doc_summary": { "type": "string" },
    "tactics": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "tactic_id": { "type": "string" },
          "tactic_name": { "type": "string" },
          "justification": { "type": "string" },
          "confidence": { "type": "number" },
          "techniques": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": { "type": "string" },
                "name": { "type": "string" },
                "confidence": { "type": "number" },
                "justification": { "type": "string" }
              },
              "required": ["id", "name", "confidence", "justification"]
            }
          }
        },
        "required": ["tactic_id", "tactic_name", "justification", "confidence", "techniques"]
      }
    }
  },
  "required": ["doc_summary", "tactics"]
}

#system instructions
    system_text = (
        "You are a cybersecurity analyst. Map intelligence reports to MITRE ATT&CK tactics and techniques. "
        "Use only the report summary and the retrieved ATT&CK context as evidence. "
        "If the report does NOT contain any cyber threat or attack-related information, "
        "return an explicit JSON message indicating that no relevant cybersecurity intelligence was found, along with a 2 sentence summary of the summary text (No Information if document has no text). "
        "Do not attempt to invent tactics or techniques in that case."
    )

    instructions = (
        "You MUST return JSON ONLY. Follow these steps strictly:\n"
        "1) Relevance check:\n"
        "   - If the report summary contains cybersecurity intelligence (threat activity, intrusion analysis, malware, IOCs, TTPs), continue.\n"
        "   - If NOT, return exactly (where the value for doc summary is a 2 sentence summary of whatever information the summary text is about): "
        "{\"message\": \"No relevant cybersecurity intelligence information detected in the provided report summary.\", \"doc_summary\": }\n\n"
        "2) When relevant, produce a JSON array of objects conforming to the following JSON Schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Additional rules:\n"
        "-A two sentence summary of the document as doc_summary is a must"
        "- Max 5 techniques/sub-techniques per tactic.\n"
        "- Confidence values in [0,1].\n"
        "- Each technique MUST include a short justification grounded in the retrieved context.\n"
        "- Output must be JSON only (no markdown, no prose)."
    )


    return {
                "system": [{"text": system_text}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": instructions},
                    {"text": "Report summary:\n" + (report_summary or "")},

                    {"text": "Retrieved context (top results):\n" + json.dumps(retrieved or [], indent=2)},
                    {"text": "Return JSON only — either a valid ATT&CK mapping array or a single message object if no intelligence is found."}
                ]
            }
        ],
        # Nova Micro / Bedrock-style knobs; keep yours as-is
        "inferenceConfig": {
            "maxTokens": 1200,
            "temperature": 0.2,
            "topP": 0.9
        }
    }

def call_nova_micro(payload: Dict[str, Any]) -> Any:
    client = bedrock_client()
    resp = client.invoke_model(
        modelId=nova_model_id,
        body=json.dumps(payload).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    body = json.loads(resp["body"].read())

    # Typical converse-like shape:
    txt = None
    try:
        txt = body["output"]["message"]["content"][0]["text"]
    except Exception:
        for k in ("outputText", "generated_text", "text"):
            if k in body:
                txt = body[k]
                break

    if not txt:
        return body

    cleaned_txt = clean_nova_response(txt)


    try:
        return json.loads(repair_json(cleaned_txt))
    except Exception as e:
        logging.exception(f"Failed to parse Nova Micro response: {cleaned_txt}")
        raise e





