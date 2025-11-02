import json
import sys
from pathlib import Path
#Shared btn both structured_mapper.py and unstructured_mapper.py
BEDROCK_REGION = 'us-east-1'
BEDROCK_REGION_AP = 'ap-southeast-2'
BEDROCK_TITAN_MODEL_ID = 'amazon.titan-embed-text-v2:0'
BEDROCK_LLM_MODEL_ID = 'us.amazon.nova-micro-v1:0'

#specifc to the structured_mapper.py
STRUCTURED_TOP_K = 8
STRUCTURED_INCLUDE_NORMAL =False
STRUCTURED_ENRICHED_DIR = 'knowledge'
STRUCTURED_INDEX_DIR = Path('index')
STRUCTURED_DOCS_JSONL = Path(STRUCTURED_ENRICHED_DIR) / 'combined_knowledge.jsonl'
STRUCTURED_PYCARET_ML_MODEL = 'classifier/netlog_pycaret_classification_model'
STRUCTURED_UNSW_COL_DEFS = Path('NUSW-NB15_features_UTF8_name_decsription.csv')

#Specifc to the unstructured_mapper.py
UNSTRUCTURED_INDEX_DIR = Path('index2')
UNSTRUCTURED_MAX_WORDS = 2500 # maximun number of word to ready from an input document summary
UNSTRUCTURED_MAX_CHUNKS = 20
UNSTRUCTURED_CHUNK_SIZE = 1200
UNSTRUCTURED_CHUNK_OVERLAP = 200
UNSTRUCTURED__TOP_K_PER_CHUNK = 5
UNSTRUCTURED_TOP_CONTEXTS = 10
UNSTRUCTURED_EMBED_WORKERS = 0 #workser when building embeddings with Titan embedding
UNSTRUCTURED_EMBED_MAX_RETRIES = 4# how many times to retry embedding before sleeping
UNSTRUCTURED_EMBED_BASE_SLEEP = 1.0 # seconds to sleep between retries. final sleep time value is: (UNSTRUCTURED_EMBED_BASE_SLEEP * 2^(attempt -1)) seconds

#Shared helpers

def info(msg): print(f"*********** [INFO] {msg}")
def warn(msg): print(f"*********** [WARN] {msg}")
def err(msg):  print(f"*********** [ERROR] {msg}", file=sys.stderr)

def save_json(data, filename):
    info(f'saving json file:{filename}')
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def clean_nova_response(response):
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned