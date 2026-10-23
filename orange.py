import os
import sys
import urllib.request
import urllib.parse
import json
from functools import reduce

# Configuration Constants
OPENFDA_BASE_URL = "https://api.fda.gov/drug/drugsfda.json"
BATCH_SIZE = 100
MAX_RECORDS = 500  # Adjust as needed (set to 500 for a healthy sample size, increase for full dataset)
OUTPUT_DIR = "./output"
RAW_FDA_FILE = os.path.join(OUTPUT_DIR, "raw_fda_drugs.json")
OUTPUT_JSON_FILE = os.path.join(OUTPUT_DIR, "enriched_drugs.json")
REPORT_FILE = os.path.join(OUTPUT_DIR, "enrichment_report.txt")

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

def get_pubchem_smiles(active_ingredient):
    """Fetch authoritative canonical SMILES from PubChem PUG REST API"""
    encoded_name = urllib.parse.quote(active_ingredient.strip())
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/json"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            properties = data.get('PropertyTable', {}).get('Properties', [])
            if properties:
                return properties[0].get('CanonicalSMILES', 'NOT_FOUND')
    except Exception:
        pass
    return 'NOT_FOUND'

def query_single_model(drug_record, model_name):
    """Query a specific Gemma model for SMILES extraction"""
    prompt = f"Provide the canonical SMILES string for the drug '{drug_record['active_ingredient']}' (Trade name: {drug_record['trade_name']}). Output ONLY the valid SMILES string or 'UNKNOWN'."
    
    print(f"\n------------------------------------------------------------")
    print(f"[Prompt sent to {model_name} for '{drug_record['active_ingredient']}']:\n{prompt}")
    print(f"------------------------------------------------------------")

    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False})
    
    try:
        req = urllib.request.Request(GEMMA_ENDPOINT, data=payload.encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            smiles = res_data.get('response', '').strip()
            
            print(f"[{model_name} Response Received]:\n{smiles}")
            print(f"------------------------------------------------------------\n")
            return smiles if smiles != 'UNKNOWN' else 'N/A'
    except Exception as e:
        print(f"[Error connecting to {model_name} at {GEMMA_ENDPOINT}]: {e}")
        print(f"------------------------------------------------------------\n")
        return 'SERVICE_UNAVAILABLE'

def enrich_record(drug_record):
    """Enrich a record with PubChem ground truth and model evaluations"""
    print(f"-> Processing: {drug_record['active_ingredient']} ({drug_record['trade_name']})")
    
    # 1. Fetch authoritative SMILES from PubChem
    pubchem_smiles = get_pubchem_smiles(drug_record['active_ingredient'])
    
    # 2. Query LLM models
    evals = {model.replace(':', '_'): query_single_model(drug_record, model) for model in MODELS}
    
    return {
        **drug_record,
        'pubchem_smiles': pubchem_smiles,
        'gemma_evaluations': evals
    }

def save_to_json(records, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print(f"-> Successfully saved JSON results to: {file_path}")

def generate_report(records, file_path):
    """Generate a comparative text report verifying LLM performance against PubChem ground truth"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    total = len(records)
    
    if total > 0:
        matches_2b = sum(1 for r in records if r['gemma_evaluations'].get('gemma_2b') == r['pubchem_smiles'] and r['pubchem_smiles'] != 'NOT_FOUND')
        matches_7b = sum(1 for r in records if r['gemma_evaluations'].get('gemma_7b') == r['pubchem_smiles'] and r['pubchem_smiles'] != 'NOT_FOUND')
        pubchem_found = sum(1 for r in records if r['pubchem_smiles'] != 'NOT_FOUND')
    else:
        matches_2b = matches_7b = pubchem_found = 0

    report_lines = [
        "==================================================",
        "       FDA DRUG ENRICHMENT & LLM BENCHMARK REPORT  ",
        "==================================================",
        f"Total Records Processed: {total}",
        f"PubChem Ground Truth Found: {pubchem_found} / {total}",
        f"Gemma 2B Exact Matches with PubChem: {matches_2b} / {total}",
        f"Gemma 7B Exact Matches with PubChem: {matches_7b} / {total}",
        "\n" + "-" * 50,
        "DETAILED COMPARISONS:",
        "-" * 50
    ]
    
    for r in records:
        report_lines.extend([
            f"Trade Name       : {r['trade_name']}",
            f"Active Ingredient: {r['active_ingredient']}",
            f"PubChem (Truth)  : {r['pubchem_smiles']}",
            f"Gemma 2B         : {r['gemma_evaluations'].get('gemma_2b')}",
            f"Gemma 7B         : {r['gemma_evaluations'].get('gemma_7b')}",
            "-" * 50
        ])
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"-> Successfully generated summary report at: {file_path}")

def run_pipeline():
    raw_results = get_or_download_fda_data()
    
    print(f"-> Parsing and filtering {len(raw_results)} raw records...")
    processed_records = list(
        map(
            clean_record,
            filter(is_valid_drug, raw_results)
        )
    )[:50]  # Limiting sample for demo; adjust slice or remove for larger batches
    
    if not processed_records:
        print("Warning: No matching drug records found from query filter.")
        return

    print(f"-> Enriching records via PubChem and Gemma models ({', '.join(MODELS)})...")
    enriched_records = list(map(enrich_record, processed_records))
    
    # Save outputs to volume paths
    save_to_json(enriched_records, OUTPUT_JSON_FILE)
    generate_report(enriched_records, REPORT_FILE)
    
    print(f"\n--- Pipeline Completed Successfully ({len(enriched_records)} records processed) ---")

if __name__ == '__main__':
    run_pipeline()