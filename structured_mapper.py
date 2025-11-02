import random
from datetime import datetime
import io
from flask import request, jsonify

from structured_helpers import *

import logging
logging.basicConfig(level=logging.INFO)

from shared_and_constants import *

def dir_navigator(filename: str) -> Path:
    current_dir = Path(__file__)

    # One level up (parent directory)
    one_up_dir = current_dir.parent.parent
    return one_up_dir / filename


def ping_mapper_handler(save_output=False):

    SAMPLE_INPUT_FILES = ['unsw_nb15_formated_network_log_tst_part01.csv','unsw_nb15_formated_network_log_tst_part02.csv','unsw_nb15_formated_network_log_tst_part03.csv','unsw_nb15_formated_network_log_tst_part04.csv']

    INPUT_CSV =  Path('static') / random.choice(SAMPLE_INPUT_FILES)

    info(f"Using input file: {INPUT_CSV}")
    logging.info(f"Using input file: {INPUT_CSV}")
    #INPUT_CSV = Path('fuzzers.csv')
    INDEX_DIR_ = Path(STRUCTURED_INDEX_DIR)

    # 1) Build class profile
    info("Building class profile...")
    profile = build_class_profile(INPUT_CSV, include_normal=INCLUDE_NORMAL)
    info("Class profile built.")

    # 2) Ensure index exists
    # builds faiss index if not present
    # ensure_index(DOCS_JSONL, INDEX_DIR, region=BEDROCK_REGION, embed_model_id=

    # 3) Retrieval of relevant chuncks based on the profile of input file
    qtext = as_query_text(profile)
    qvec = titan_embed_single(qtext, region=BEDROCK_REGION, model_id=BEDROCK_TITAN_MODEL_ID)
    retrieved = retrieve(INDEX_DIR_, qvec, topk=STRUCTURED_TOP_K)
    info(f"Retrieved {len(retrieved)} chunks.")

    # 4) Nova Micro
    info("Calling Nova Micro to generate final JSON...")
    prompt = build_llm_messages(profile, retrieved)
    llm_json = call_nova_micro(prompt, region=BEDROCK_REGION, model_id=BEDROCK_LLM_MODEL_ID)

    # 5) Save
    info({
        "results": llm_json,
        "retrieval_context": retrieved,
        "_meta": {
            "csv": str(INPUT_CSV),
            "region": BEDROCK_REGION,
            "embed_model_id": BEDROCK_TITAN_MODEL_ID,
            "llm_model_id": BEDROCK_LLM_MODEL_ID,
            "topk": STRUCTURED_TOP_K
        }
    })

    result = {
        "results": llm_json,
        "retrieval_context": retrieved
    }

    if save_output:
        save_json(result,f'{datetime.today().strftime("%Y%m%d_%H%M%S")}llm_output.json' )


    info("Returning result .")
    return jsonify(result)



def cloud_mapper(df:DataFrame):
    INDEX_DIR_ = STRUCTURED_INDEX_DIR

    # 1) Build class profile
    info("Building class profile...")
    profile = build_class_profile_df(df, include_normal=INCLUDE_NORMAL)

    # 2) Ensure index exists/ build index (optional)

    # 3) Retrieval of relevant chuncks based on the profile of input file
    qtext = as_query_text(profile)
    qvec = titan_embed_single(qtext, region=BEDROCK_REGION, model_id=BEDROCK_TITAN_MODEL_ID)
    retrieved = retrieve(INDEX_DIR_, qvec, topk=STRUCTURED_TOP_K)
    info(f"Retrieved {len(retrieved)} chunks.")

    # 4) Nova Micro
    info("Calling Nova Micro to generate final JSON...")
    prompt = build_llm_messages(profile, retrieved)
    llm_json = call_nova_micro(prompt, region=BEDROCK_REGION, model_id=BEDROCK_LLM_MODEL_ID)

    # 5) Return Result

    return jsonify({
        "results": llm_json,
        "retrieval_context": retrieved
    })

def run_cloud_csv_mapper():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"status": "error", "message": "No file"}), 400
    try:
        df_csv = pd.read_csv(io.BytesIO(uploaded_file.read()))
        return cloud_mapper(df_csv)
    except Exception as e:
        logging.exception("Error in /run_cloud_csv_mapper: (map-csv)")
        return jsonify({"status": "error", "message": str(e)}), 400




if __name__ == '__main__':
    print(ping_mapper_handler(save_output=True))

