import os
import sys
import urllib.request
import json
from functools import reduce

# Configuration Constants
OPENFDA_BASE_URL = "https://api.fda.gov/drug/drugsfda.json"
BATCH_SIZE = 100
MAX_RECORDS = 5000  # Adjust as needed
OUTPUT_DIR = "./output"
RAW_FDA_FILE = os.path.join(OUTPUT_DIR, "raw_fda_drugs.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "enriched_drugs.json")

# Dynamic endpoint resolution for Docker vs Host execution
GEMMA_ENDPOINT = os.environ.get('GEMMA_ENDPOINT', 'http://localhost:11434/api/generate')
MODELS = ["gemma:2b", "gemma:7b"]

def fetch_all_openfda_data():
    """Fetch complete drug database from openFDA API with pagination"""
    all_results = []
    skip = 0
    
    print("-> Fetching complete drug database from openFDA API (paginated)...")
    while skip < MAX_RECORDS:
        url = f"{OPENFDA_BASE_URL}?limit={BATCH_SIZE}&skip={skip}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))
                results = data.get('results', [])
                if not results:
                    break
                all_results.extend(results)
                print(f"   Fetched {len(all_results)} records so far...")
                if len(results) < BATCH_SIZE:
                    break
                skip += BATCH_SIZE
        except Exception as e:
            print(f"Error fetching batch at skip {skip}: {e}", file=sys.stderr)
            break
            
    return all_results

def get_or_download_fda_data():
    """Check volume cache for raw FDA records before downloading"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(RAW_FDA_FILE):
        print(f"-> Found cached raw FDA data at {RAW_FDA_FILE}. Skipping download.")
        with open(RAW_FDA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    raw_data = fetch_all_openfda_data()
    print(f"-> Saving raw FDA data to volume cache: {RAW_FDA_FILE}")
    with open(RAW_FDA_FILE, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f)
    return raw_data

# Reliable filtering using core 'products' array schema
is_valid_drug = lambda record: bool(
    record.get('products') and 
    record['products'][0].get('brand_name') and 
    record['products'][0].get('active_ingredients')
)

clean_record = lambda record: {
    'trade_name': record['products'][0]['brand_name'].strip().title(),
    'active_ingredient': record['products'][0]['active_ingredients'][0]['name'].strip(),
    'appl_no': record.get('application_number', '').strip()
}

def query_single_model(drug_record, model_name):
    """Query a specific Gemma model for SMILES extraction"""
    prompt = f"Provide the canonical SMILES string for the drug '{drug_record['active_ingredient']}' (Trade name: {drug_record['trade_name']}). Output ONLY the valid SMILES string or 'UNKNOWN'."
    
    print(f"\n------------------------------------------------------------")
    print(f"[Prompt sent to {model_name} for '{drug_record['active_ingredient']}']:\n{prompt}")
    print(f"------------------------------------------------------------")

    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False})
    
    try:
        req = urllib.request.Request(GEMMA_ENDPOINT, data=payload.encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=90) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            smiles = res_data.get('response', '').strip()
            
            print(f"[{model_name} Response Received]:\n{smiles}")
            print(f"------------------------------------------------------------\n")
            return smiles if smiles != 'UNKNOWN' else 'N/A'
    except Exception as e:
        print(f"[Error connecting to {model_name} at {GEMMA_ENDPOINT}]: {e}")
        print(f"------------------------------------------------------------\n")
        return 'SERVICE_UNAVAILABLE'

def enrich_record_with_models(drug_record):
    """Query all configured Gemma models and tabulate results into the record"""
    results_map = {model.replace(':', '_'): query_single_model(drug_record, model) for model in MODELS}
    return {
        **drug_record,
        'gemma_evaluations': results_map
    }

def save_to_file(records, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print(f"-> Successfully saved results to host-accessible file: {file_path}")
    return records

def run_pipeline():
    raw_results = get_or_download_fda_data()
    
    print(f"-> Parsing and filtering {len(raw_results)} raw records...")
    processed_records = list(
        map(
            clean_record,
            filter(is_valid_drug, raw_results)
        )
    )
    
    if not processed_records:
        print("Warning: No matching drug records found from query filter.")
        return

    print(f"-> Querying Gemma models ({', '.join(MODELS)}) via {GEMMA_ENDPOINT}...")
    enriched_records = list(map(enrich_record_with_models, processed_records))
    
    save_to_file(enriched_records, OUTPUT_FILE)
    print(f"\n--- Final Enriched Drug Mappings Completed ({len(enriched_records)} records processed) ---")

if __name__ == '__main__':
    run_pipeline()