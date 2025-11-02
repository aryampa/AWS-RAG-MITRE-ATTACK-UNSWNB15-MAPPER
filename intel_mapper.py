from flask import jsonify, request

from intel_mapper_helpers import *


def analyze_static():
    """
    Run unstructured intelligence mapping on a fixed static PDF.

    Steps:
        1. Load the predefined PDF file from the disk.
        2. Extract text using pdfminer low-level API.
        3. Chunk text into overlapping windows.
        4. Retrieve top-k MITRE ATT&CK docs via FAISS search.
        5. Build a Nova Micro LLM prompt with summary + context.
        6. Call Nova Micro and return results.

    Returns:
        flask.Response: JSON with keys:
            - retrieval: Retrieved ATT&CK context
            - llm_result: Model-generated tactic/technique mappings

    Justification:
        - Provides a quick testable pipeline without user uploads.
        - Validates end-to-end workflow (PDF → embeddings → retrieval → LLM).
    """

    static_file = Path("static") /'threat_Intelligence_sample.pdf'

    pdf_bytes = None

    with open(static_file, 'rb') as file:
        pdf_bytes = file.read()

    return analysis_engine(pdf_bytes)



def analyse():

    """
    Run unstructured intelligence mapping on a user-uploaded PDF.

    Steps:
        1. Read PDF from request.
        2. Extract text and validate it.
        3. Chunk into overlapping segments.
        4. Embed and retrieve relevant ATT&CK docs.
        5. Build and send Nova Micro prompt.
        6. Return structured JSON result.

    Returns:
        flask.Response: JSON with retrieval + LLM result.

    Raises:
        400 Bad Request if:
            - File is missing
            - File is empty
            - Text extraction or chunking fails

    Justification:
        - Enables real-world use of pipeline with arbitrary reports.
        - Mirrors static workflow but handles external inputs safely.
    """

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()
    return analysis_engine(pdf_bytes)

def analysis_engine(pdf_bytes):
    if not pdf_bytes:
        return jsonify({"error": "empty file"}), 400

    report_text = extract_pdf_text(pdf_bytes)
    if not report_text:
        return jsonify({"error": "could not extract text"}), 400

    # report summary is the first UNSTRUCTURED_MAX_WORDS words of the report. This is to exert control over latency incase reports with alot of text
    report_summary = report_text if (len(report_text) >= UNSTRUCTURED_MAX_WORDS) else report_text[
        :UNSTRUCTURED_MAX_WORDS]
    chunks = chunk_text(report_text, size=chunk_size, overlap=chunk_overlap, limit=max_chunks)
    if not chunks:
        return jsonify({"error": "no text chunks created"}), 400

    retrieval = retrieve_for_chunks(chunks)
    prompt = build_nova_prompt(retrieval, report_summary)
    llm_json = call_nova_micro(prompt)

    return jsonify({
        "retrieval_context": retrieval,
        "results": llm_json
    })

if __name__ == '__main__':
    print(analyze_static())
