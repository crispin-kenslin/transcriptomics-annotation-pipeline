"""
OPTIMIZED TRANSCRIPTOMICS ANALYSIS WEB SERVER WITH UNIFIED PROCESSING
"""

import os
import time
import json
import base64
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib.parse import quote
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
import community
import warnings
import re
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')


def env_float(key, default):
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def env_bool(key, default):
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_EXPR_FILE = "GEO_data.csv"
DEFAULT_OUTPUT_DIR = "pipeline_results"
DEFAULT_LOG2FC_THRESHOLD = 1.0
DEFAULT_PVALUE_THRESHOLD = 0.05
DEFAULT_DISTANCE_PERCENTILE = 95

EXPR_FILE = os.getenv("EXPR_FILE", DEFAULT_EXPR_FILE)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
CACHE_DIR = os.getenv("CACHE_DIR", f"{OUTPUT_DIR}/cache")
GENE_DIR = os.getenv("GENE_DIR", f"{OUTPUT_DIR}/genes")
UP_GENES_DIR = os.getenv("UP_GENES_DIR", f"{GENE_DIR}/upregulated")
DOWN_GENES_DIR = os.getenv("DOWN_GENES_DIR", f"{GENE_DIR}/downregulated")
NETWORK_DIR = os.getenv("NETWORK_DIR", f"{OUTPUT_DIR}/networks")
UP_NETWORK = os.getenv("UP_NETWORK", f"{NETWORK_DIR}/upregulated")
DOWN_NETWORK = os.getenv("DOWN_NETWORK", f"{NETWORK_DIR}/downregulated")
up_network_img = f"{UP_NETWORK}/upregulated_network.svg"
down_network_img = f"{DOWN_NETWORK}/downregulated_network.svg"

LOG2FC_THRESHOLD = env_float("LOG2FC_THRESHOLD", DEFAULT_LOG2FC_THRESHOLD)
PVALUE_THRESHOLD = env_float("PVALUE_THRESHOLD", DEFAULT_PVALUE_THRESHOLD)
DISTANCE_PERCENTILE = env_float("DISTANCE_PERCENTILE", DEFAULT_DISTANCE_PERCENTILE)
USE_ADJUSTED_PVALUE = env_bool("USE_ADJUSTED_PVALUE", False)
RUN_DASH_SERVER = env_bool("RUN_DASH_SERVER", True)

# API Endpoints
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/files"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
ENSEMBL_API = "https://rest.ensembl.org"
STRING_API = "https://string-db.org/api"
STRING_SPECIES = 39947
EGGNOG_API = "http://eggnogapi5.embl.de/nog_data/json"
INTERPRO_API = "https://www.ebi.ac.uk/interpro/api/entry/interpro"
KEGG_API = "https://rest.kegg.jp"
ALPHASYNC_BASE = "https://alphasync.stjude.org/api/v1"
PHOBIUS_BASE = "https://www.ebi.ac.uk/Tools/services/rest/phobius"
PHOBIUS_EMAIL = os.getenv("PHOBIUS_EMAIL", "wp-angular-web@ebi.ac.uk")
GOR4_POST_URL = "https://npsa-prabi.ibcp.fr/cgi-bin/secpred_gor4.pl"
GOR4_TMP_BASE = "https://npsa-prabi.ibcp.fr/tmp"
GOR4_ALI_WIDTH = os.getenv("GOR4_ALI_WIDTH", "70")
GOR4_MAX_POLLS = int(os.getenv("GOR4_MAX_POLLS", "20"))
GOR4_POLL_DELAY = env_float("GOR4_POLL_DELAY", 2.0)
GOR4_HEADER = "Pos|AA|Prediction|P(H)|P(E)|P(C)"
GOR4_JOB_REGEX = re.compile(r"/tmp/([a-f0-9]+)\.gor4")

# Create directories
for d in [OUTPUT_DIR, CACHE_DIR, GENE_DIR, UP_GENES_DIR, DOWN_GENES_DIR, 
          NETWORK_DIR, UP_NETWORK, DOWN_NETWORK]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("TRANSCRIPTOMICS ANALYSIS PIPELINE - UNIFIED VERSION")
print("=" * 60)

# ==========================================================
# CACHING SYSTEM
# ==========================================================
def load_cache(cache_file):
    cache_path = f"{CACHE_DIR}/{cache_file}"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache_file, data):
    cache_path = f"{CACHE_DIR}/{cache_file}"
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def is_step_complete_with_files(step_name, required_files):
    marker = f"{CACHE_DIR}/{step_name}.done"
    if not os.path.exists(marker):
        return False
    for file_path in required_files:
        if not os.path.exists(file_path):
            return False
    return True

def mark_step_complete(step_name):
    marker = f"{CACHE_DIR}/{step_name}.done"
    with open(marker, 'w') as f:
        f.write(str(time.time()))

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def create_session():
    session = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=0.5,
                  status_forcelist=(500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

SESSION = create_session()

def safe_request(url, params=None, timeout=30):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        if r.ok and r.text.strip():
            return r
    except Exception as e:
        pass
    return None

def safe_json(response):
    if not response:
        return None
    try:
        return response.json()
    except:
        return None

def download_file(url, path, binary=True):
    try:
        r = safe_request(url, timeout=30)
        if r and len(r.content) > 100:
            with open(path, "wb" if binary else "w") as f:
                f.write(r.content if binary else r.text)
            return True
    except:
        pass
    return False


def extract_sequence_from_fasta(fasta_text):
    lines = [ln.strip() for ln in fasta_text.splitlines() if ln.strip()]
    seq = "".join([ln for ln in lines if not ln.startswith(">")])
    return re.sub(r"[^A-Za-z]", "", seq).upper()


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_protein_sequence(seq):
    """Keep only standard/ambiguous amino-acid codes accepted by ESMFold."""
    if not seq:
        return ""
    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    return "".join([aa for aa in seq if aa in VALID_AA])


def _gor4_wait_for_file(url, binary=True):
    for _ in range(GOR4_MAX_POLLS):
        try:
            resp = SESSION.get(url, timeout=60)
            if resp.status_code == 200 and resp.content:
                return resp.content if binary else resp.text
        except Exception:
            pass
        time.sleep(GOR4_POLL_DELAY)
    return None


def predict_secondary_structure_gor4(seq, gene_dir=None, gene_id=None):
    seq = clean_protein_sequence(seq)
    if not seq:
        return None

    payload = {"notice": seq, "ali_width": GOR4_ALI_WIDTH}
    try:
        resp = SESSION.post(GOR4_POST_URL, data=payload, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"    GOR4 submission failed: {exc}")
        return None

    match = GOR4_JOB_REGEX.search(resp.text)
    if not match:
        print("    GOR4 job ID missing in response")
        return None

    job_id = match.group(1)
    base_url = f"{GOR4_TMP_BASE}/{job_id}.gor4"
    reports_dir = Path(gene_dir) if gene_dir else Path(CACHE_DIR) / "gor4"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = gene_id if gene_id else f"gor4_{job_id}"

    artifact_paths = {}
    state_bytes = _gor4_wait_for_file(f"{base_url}.mpsa_state.gif", binary=True)
    if state_bytes:
        state_path = reports_dir / f"{artifact_prefix}_mpsa_state.gif"
        with open(state_path, "wb") as fh:
            fh.write(state_bytes)
        artifact_paths["gor4_state"] = str(state_path)

    profile_bytes = _gor4_wait_for_file(f"{base_url}.mpsa1.gif", binary=True)
    if profile_bytes:
        profile_path = reports_dir / f"{artifact_prefix}_mpsa1.gif"
        with open(profile_path, "wb") as fh:
            fh.write(profile_bytes)
        artifact_paths["gor4_profile"] = str(profile_path)

    result_text = _gor4_wait_for_file(base_url, binary=False)
    if not result_text:
        print("    GOR4 result text was not ready in time")
        return None

    raw_path = reports_dir / f"{artifact_prefix}_gor4_raw.txt"
    try:
        with open(raw_path, "w") as fh:
            fh.write(result_text)
        artifact_paths["gor4_raw"] = str(raw_path)
    except Exception as exc:
        print(f"    Failed to save raw GOR4 output: {exc}")

    rows = []
    for line in result_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("SEQ"):
            continue
        parts = stripped.split()
        pos = None
        if len(parts) == 6 and parts[0].isdigit():
            pos = int(parts[0])
            aa, pred = parts[1], parts[2]
            probs = parts[3:6]
        elif len(parts) == 5 and len(parts[0]) == 1:
            aa, pred = parts[0], parts[1]
            probs = parts[2:5]
        else:
            continue

        try:
            prob_h = float(probs[0])
            prob_e = float(probs[1])
            prob_c = float(probs[2])
        except ValueError:
            continue

        rows.append({
            "pos": pos if pos is not None else len(rows) + 1,
            "aa": aa,
            "pred": pred,
            "prob_h": prob_h,
            "prob_e": prob_e,
            "prob_c": prob_c,
        })

    if not rows:
        print("    GOR4 output contained no residue rows")
        return None

    states = "".join([row["pred"] for row in rows])
    confidence = []
    formatted_lines = [
        gene_id or artifact_prefix,
        str(len(seq)),
        GOR4_HEADER,
    ]

    for row in rows:
        pred = row["pred"]
        prob_map = {"H": row["prob_h"], "E": row["prob_e"], "C": row["prob_c"]}
        confidence.append(prob_map.get(pred, 0.0))
        formatted_lines.append(
            f"{row['pos']}|{row['aa']}|{pred}|{row['prob_h']:.3f}|{row['prob_e']:.3f}|{row['prob_c']:.3f}"
        )

    return {
        "states": states,
        "confidence": confidence,
        "formatted_lines": formatted_lines,
        "method": "gor4",
        "gor4_job_id": job_id,
        "gor4_artifacts": artifact_paths,
    }


def predict_secondary_structure(seq, acc=None, gene_dir=None, gene_id=None):
    """Prefer GOR4 API, then AlphaSync, finally a heuristic fallback."""
    seq = clean_protein_sequence(seq)
    if not seq:
        return {"states": "", "confidence": [], "formatted_lines": [], "method": "none"}

    gor4_result = predict_secondary_structure_gor4(seq, gene_dir=gene_dir, gene_id=gene_id)
    if gor4_result and gor4_result.get("states"):
        return gor4_result

    def _fetch_by_acc(accession):
        if not accession:
            return None
        try:
            resp = safe_request(f"{ALPHASYNC_BASE}/protein/{accession}", timeout=30)
            data = safe_json(resp)
            if not data or not isinstance(data, list):
                return None
            if not data:
                return None
            states = []
            conf = []
            for res in data:
                sec = res.get("sec") or "C"
                states.append(sec)
                plddt_val = res.get("plddt")
                if isinstance(plddt_val, (int, float)):
                    conf.append(float(plddt_val))
            return {"states": "".join(states), "confidence": conf, "formatted_lines": [], "method": "alphasync"}
        except Exception:
            return None

    # 1) If we have a UniProt accession, query directly
    remote = _fetch_by_acc(acc)
    if remote and remote.get("states"):
        return remote

    # 2) If no accession or lookup failed, try mapping the sequence when URL length is safe
    if seq and len(seq) <= 2500:  # avoid overly long URLs
        try:
            resp_map = safe_request(f"{ALPHASYNC_BASE}/mapseq/{quote(seq)}", timeout=30)
            map_data = safe_json(resp_map)
            if map_data and isinstance(map_data, list) and map_data:
                mapped_acc = (
                    map_data[0].get("acc")
                    or map_data[0].get("map")
                    or map_data[0].get("value")
                )
                remote = _fetch_by_acc(mapped_acc)
                if remote and remote.get("states"):
                    return remote
        except Exception:
            pass

    # 3) Fallback: heuristic so we always have a track
    return predict_secondary_structure_local(seq)


def predict_secondary_structure_local(seq):
    """Lightweight PSIPRED-like heuristic so we always return a track."""
    if not seq:
        return {"states": "", "confidence": [], "formatted_lines": [], "method": "heuristic"}

    helix_pref = set("AILMFWV")
    strand_pref = set("VIFYWTC")
    states = []
    conf = []
    for aa in seq:
        if aa in helix_pref:
            states.append("H")
            conf.append(0.78)
        elif aa in strand_pref:
            states.append("E")
            conf.append(0.7)
        else:
            states.append("C")
            conf.append(0.45)
    return {"states": "".join(states), "confidence": conf, "formatted_lines": [], "method": "heuristic"}


def run_esmf_fold(seq, out_path):
    """Disabled: no reliable public fold API available."""
    return None

# Initialize caches
annotation_cache = load_cache("annotations.json")
uniprot_cache = load_cache("uniprot_search.json")
eggnog_cache = load_cache("eggnog.json")
kegg_cache = load_cache("kegg.json")
alphafold_cache = load_cache("alphafold.json")
uniprot_entry_cache = load_cache("uniprot_entry.json")
string_tsv_cache = load_cache("string_tsv.json")
string_cytoscape_cache = {}
gene_panel_cache = {}




####Timer###
start_time = time.time()
############
# ==========================================================
# DATA LOADING
# ==========================================================
print("\n[STEP 1/4] Loading expression data...")

if is_step_complete_with_files("data_loading", [
    f"{CACHE_DIR}/processed_data.csv",
    f"{OUTPUT_DIR}/upregulated_genes.csv",
    f"{OUTPUT_DIR}/downregulated_genes.csv"
]):
    print("  ✓ Using cached data")
    df = pd.read_csv(f"{CACHE_DIR}/processed_data.csv")
    upregulated = pd.read_csv(f"{OUTPUT_DIR}/upregulated_genes.csv")
    downregulated = pd.read_csv(f"{OUTPUT_DIR}/downregulated_genes.csv")
else:

    print("  Processing expression data...")
    df = pd.read_csv(EXPR_FILE)
    required_cols = ["gene_id", "log2FC", "p_value"]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=required_cols)
    df = df[df["p_value"] > 0]

    # Low-expression/low-signal filter: remove genes with very low absolute log2FC and very high p-value
    # (If you have a mean/total count column, use it here. Otherwise, this is a proxy.)
    min_abs_log2fc = 0.1  # Remove genes with almost no change
    max_pval = 0.99       # Remove genes with no evidence of change
    df = df[(df["log2FC"].abs() >= min_abs_log2fc) & (df["p_value"] <= max_pval)]

    if df.empty:
        raise ValueError("No valid data after low-expression filtering")

    # Calculate FDR-adjusted p-values for all genes
    _, adj_pvals, _, _ = multipletests(df["p_value"], method='fdr_bh')
    df["adj_p_value"] = adj_pvals
    # Save upload.csv with FDR column for all genes
    upload_path = os.path.join(OUTPUT_DIR, "upload.csv")
    df_upload = df.copy()
    df_upload.to_csv(upload_path, index=False)

    pval_column = "adj_p_value" if USE_ADJUSTED_PVALUE else "p_value"
    df["neg_log10_p"] = -np.log10(df[pval_column])
    df["euclidean_distance"] = np.sqrt(df["log2FC"]**2 + df["neg_log10_p"]**2)

    cutoff = np.percentile(df["euclidean_distance"], DISTANCE_PERCENTILE)
    outliers = df[df["euclidean_distance"] >= cutoff]

    upregulated = outliers[
        (outliers["log2FC"] >= LOG2FC_THRESHOLD) &
        (outliers[pval_column] <= PVALUE_THRESHOLD)
    ].copy()

    downregulated = outliers[
        (outliers["log2FC"] <= -LOG2FC_THRESHOLD) &
        (outliers[pval_column] <= PVALUE_THRESHOLD)
    ].copy()

    df.to_csv(f"{CACHE_DIR}/processed_data.csv", index=False)
    upregulated.to_csv(f"{OUTPUT_DIR}/upregulated_genes.csv", index=False)
    downregulated.to_csv(f"{OUTPUT_DIR}/downregulated_genes.csv", index=False)
    mark_step_complete("data_loading")

print(f"  Total: {len(df)} | Up: {len(upregulated)} | Down: {len(downregulated)}")

# ==========================================================
# API FUNCTIONS
# ==========================================================
def uniprot_search(gene_id, cache):
    if gene_id in cache:
        return cache[gene_id]
    
    queries = [
        f"(gene_exact:{gene_id}) AND (organism_id:39947)",
        f"(gene:{gene_id}) AND (organism_id:39947)",
        f"{gene_id} AND (organism_id:39947)"
    ]
    
    for query in queries:
        params = {"query": query, "format": "json", "size": 1}
        
        r = safe_request(UNIPROT_SEARCH, params=params, timeout=30)
        data = safe_json(r)
        if data and "results" in data and data["results"]:
            acc = data["results"][0].get("primaryAccession")
            cache[gene_id] = acc
            return acc
    
    cache[gene_id] = None
    return None

def uniprot_entry(acc):
    if acc in uniprot_entry_cache:
        return uniprot_entry_cache[acc]

    r = safe_request(f"{UNIPROT_ENTRY}/{acc}.json", timeout=30)
    data = safe_json(r)
    uniprot_entry_cache[acc] = data
    return data

def ensembl_lookup(gene_id):
    
    r = safe_request(f"{ENSEMBL_API}/lookup/id/{gene_id}?expand=1", timeout=30)
    return safe_json(r)

def get_string_tsv(gene_id, species=39947):
    cache_key = f"{gene_id}_{species}"
    if cache_key in string_tsv_cache:
        return string_tsv_cache[cache_key]

    url = f"{STRING_API}/tsv/network?identifiers={gene_id}&species={STRING_SPECIES}"
    params = {"required_score": 400}
    r_net = safe_request(url, params=params, timeout=30)

    if not r_net or not r_net.text.strip():
        string_tsv_cache[cache_key] = None
        return None

    string_tsv_cache[cache_key] = r_net.text
    return r_net.text


def get_kegg_pathways(uniprot_id):
    """
    KEGG logic (CORRECT):
    UniProt -> KO -> Pathway
    KEGG gene IDs are metadata only
    """
    cache_key = f"kegg_{uniprot_id}"
    if cache_key in kegg_cache:
        return kegg_cache[cache_key]

    results = []

    # Step 1: UniProt -> KO
    kos = uniprot_to_ko(uniprot_id)
    if not kos:
        kegg_cache[cache_key] = results
        return results

    # Step 2: UniProt -> KEGG gene IDs (optional metadata)
    kegg_genes = uniprot_to_kegg_genes(uniprot_id)

    # Step 3: KO -> Pathway
    for ko in kos[:5]:  # safety limit
        
        pathways = ko_to_pathways(ko)

        for pathway_id in pathways:
            
            r = safe_request(f"{KEGG_API}/get/{pathway_id}", timeout=30)
            if not r:
                continue

            info = parse_kegg_pathway(r.text, pathway_id)

            results.append({
                "KO": ko,
                "KEGG_gene_ids": ";".join(kegg_genes),
                "KEGG_pathway_id": pathway_id,
                "KEGG_pathway_name": info["name"],
                "KEGG_pathway_class": info["class"],
                "KEGG_description": info["description"]
            })

    kegg_cache[cache_key] = results
    return results


def parse_kegg_pathway(kegg_text, pathway_id):
    """Parse KEGG flat file format"""
    lines = kegg_text.strip().split('\n')
    info = {
        "name": "Unknown",
        "class": "Unknown",
        "description": ""
    }
    
    current_field = None
    for line in lines:
        if line.startswith('NAME'):
            parts = line.split(maxsplit=1)
            info["name"] = parts[1].strip() if len(parts) > 1 else "Unknown"
        elif line.startswith('CLASS'):
            parts = line.split(maxsplit=1)
            info["class"] = parts[1].strip() if len(parts) > 1 else "Unknown"
        elif line.startswith('DESCRIPTION'):
            current_field = 'description'
            desc = line.split(maxsplit=1)
            if len(desc) > 1:
                info["description"] = desc[1].strip()
        elif current_field == 'description' and line.startswith(' '):
            info["description"] += " " + line.strip()
        elif not line.startswith(' '):
            current_field = None
    
    return info
def uniprot_to_kegg_genes(uniprot_id):
    """
    UniProt -> KEGG gene IDs (osa / dosa)
    This is the missing link in the main pipeline
    """
    r = safe_request(f"{KEGG_API}/conv/genes/uniprot:{uniprot_id}", timeout=30)

    if not r or not r.text.strip():
        return []

    return [line.split("\t")[1] for line in r.text.strip().split("\n")]


def kegg_gene_to_kos(kegg_gene):
    r = safe_request(f"{KEGG_API}/link/ko/{kegg_gene}", timeout=30)

    if not r or not r.text.strip():
        return []

    return list(set(line.split("\t")[1] for line in r.text.strip().split("\n")))


def kegg_gene_to_pathways(kegg_gene):
    r = safe_request(f"{KEGG_API}/link/pathway/{kegg_gene}", timeout=30)

    if not r or not r.text.strip():
        return []

    return list(set(line.split("\t")[1] for line in r.text.strip().split("\n")))

def uniprot_to_ko(uniprot_id):
    """
    UniProt -> KO (authoritative KEGG mapping)
    """
    url = f"{KEGG_API}/link/ko/uniprot:{uniprot_id}"
    r = safe_request(url, timeout=30)

    if r.status_code != 200 or not r.text.strip():
        return []

    return list(set(line.split("\t")[1] for line in r.text.strip().split("\n")))


def ko_to_pathways(ko_id):
    """
    KO -> KEGG pathways
    """
    url = f"{KEGG_API}/link/pathway/{ko_id}"
    r = safe_request(url, timeout=30)

    if r.status_code != 200 or not r.text.strip():
        return []

    return list(set(line.split("\t")[1] for line in r.text.strip().split("\n")))

def get_eggnog_from_uniprot(uniprot_id):
    """
    Direct EggNOG lookup using UniProt ID
    """
    url = f"{EGGNOG_API}/protein/{uniprot_id}"
    r = safe_request(url, timeout=30)
    data = safe_json(r)
    return data

def get_eggnog_ids_from_uniprot_json(uni_json):
    eggnog_ids = []
    for xref in uni_json.get("uniProtKBCrossReferences", []):
        if xref.get("database", "").lower() == "eggnog":
            nog_id = xref.get("id")
            if nog_id:
                eggnog_ids.append(nog_id)
    return list(set(eggnog_ids))


def get_eggnog_annotation(nog_id, attribute):
    """Get EggNOG annotation for specific attribute"""
    cache_key = f"{nog_id}_{attribute}"
    if cache_key in eggnog_cache:
        return eggnog_cache[cache_key]
    
    
    url = f"{EGGNOG_API}/{attribute}/{nog_id}"
    r = safe_request(url, timeout=30)
    data = safe_json(r)
    eggnog_cache[cache_key] = data
    return data


def get_eggnog_orthologs(nog_id, tax_scope="9606,10090,559292,6239,7227,3702"):
    cache_key = f"{nog_id}_orthologs"
    if cache_key in eggnog_cache:
        return eggnog_cache[cache_key]
    try:
        url = f"{EGGNOG_API}/orthologs/{nog_id}"
        r = safe_request(url, params={"species": tax_scope}, timeout=30)
        data = safe_json(r)
        eggnog_cache[cache_key] = data
        return data
    except Exception:
        eggnog_cache[cache_key] = None
        return None

def download_alphafold_pdb(uniprot_id, out_path):
    pdb_url = f"{ALPHAFOLD_BASE}/AF-{uniprot_id}-F1-model_v6.pdb"
    success = download_file(pdb_url, out_path, binary=True)
    if success:
        return {
            "pdb_url": pdb_url,
            "model_version": "v6",
        }
    return None


def get_alphafold_info(uniprot_id):
    """Get AlphaFold metadata and structure information"""
    cache_key = f"af_{uniprot_id}"
    if cache_key in alphafold_cache:
        return alphafold_cache[cache_key]
    
    
    r = safe_request(f"{ALPHAFOLD_API}/{uniprot_id}", timeout=30)
    data = safe_json(r)
    
    if data and len(data) > 0:
        af_info = data[0]
        result = {
            "model_version": af_info.get("latestVersion", af_info.get("modelVersion", "Unknown")),
            "confidence": af_info.get("globalMetricValue", "N/A"),
            "model_page": f"{ALPHAFOLD_BASE}/entry/{uniprot_id}",
            "pdb_url": af_info.get("pdbUrl", ""),
            "created_date": af_info.get("modelCreatedDate", "Unknown")
        }
        alphafold_cache[cache_key] = result
        return result
    
    alphafold_cache[cache_key] = None
    return None

def get_string_network_cytoscape(gene_id, species=39947):
    """Get STRING network data formatted for Cytoscape"""
    cache_key = f"{gene_id}_{species}"
    if cache_key in string_cytoscape_cache:
        return string_cytoscape_cache[cache_key]

    api_url = f"{STRING_API}/tsv/network"
    params = {"identifiers": gene_id, "species": species, "required_score": 400}
    
    r = safe_request(api_url, params=params, timeout=30)
    
    if not r or not r.text.strip():
        return [], []
    
    lines = r.text.strip().split("\n")[1:]
    if not lines:
        return [], []
    
    # Build NetworkX graph
    G = nx.Graph()
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 6:
            protein1, protein2, score = parts[2], parts[3], float(parts[5])
            G.add_edge(protein1, protein2, weight=score)
    
    if len(G.edges) == 0:
        return [], []
    
    # Detect clusters
    partition = community.best_partition(G)
    
    cluster_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
    ]
    
    # Find query node
    query_node_id = None
    gene_id_upper = gene_id.upper()

    for node in G.nodes():
        node_upper = node.upper()

        if gene_id_upper in node_upper:
            query_node_id = node
            break


    
    # Prepare nodes
    nodes = []
    for node in G.nodes():
        cluster_id = partition.get(node, 0)

        if node == query_node_id:
            node_color = "#0000FF"
            node_size = 40
            font_size = 12
        else:
            node_color = cluster_colors[cluster_id % len(cluster_colors)]
            node_size = 25
            font_size = 9

        nodes.append({
            'data': {
                'id': node,
                'label': node
            },
            'style': {
                'background-color': node_color,
                'width': node_size,
                'height': node_size,
                'font-size': font_size,
                'border-width': 1,
                'border-color': '#000000' if node == query_node_id else '#555555'
            }
        })
    
    # Prepare edges
    edges = [{'data': {'source': u, 'target': v, 'weight': d['weight']}} 
             for u, v, d in G.edges(data=True)]
    
    string_cytoscape_cache[cache_key] = (nodes, edges)
    return nodes, edges

# ==========================================================
# UNIFIED GENE ANNOTATION
# ==========================================================
print("\n[STEP 2/4] Annotating genes with unified processing...")

if is_step_complete_with_files("unified_annotation", [
    f"{OUTPUT_DIR}/annotations.csv",
    f"{OUTPUT_DIR}/kegg_pathways.csv",
    f"{OUTPUT_DIR}/eggnog_go_terms.csv",
    f"{OUTPUT_DIR}/eggnog_domains.csv",
    f"{OUTPUT_DIR}/interpro_domains.csv"
]):
    print("  ✓ Using cached annotations")
    ann_df = pd.read_csv(f"{OUTPUT_DIR}/annotations.csv")
    kegg_df = pd.read_csv(f"{OUTPUT_DIR}/kegg_pathways.csv")
    eggnog_go_df = pd.read_csv(f"{OUTPUT_DIR}/eggnog_go_terms.csv")
    eggnog_domains_df = pd.read_csv(f"{OUTPUT_DIR}/eggnog_domains.csv")
    interpro_df = pd.read_csv(f"{OUTPUT_DIR}/interpro_domains.csv")
else:
    print("  Processing all annotations...")
    annotation_records = []
    kegg_records = []
    eggnog_go_records = []
    eggnog_domain_records = []
    interpro_records = []

    all_genes = pd.concat([upregulated, downregulated])
    total_genes = len(all_genes)

    for idx, (_, row) in enumerate(all_genes.iterrows(), 1):
        gene_id = row["gene_id"]
        log2fc = row["log2FC"]
        pvalue = row["p_value"]
        
        base_dir = UP_GENES_DIR if log2fc > 0 else DOWN_GENES_DIR
        gene_dir = f"{base_dir}/{gene_id}"
        os.makedirs(gene_dir, exist_ok=True)
        
        print(f"  [{idx}/{total_genes}] {gene_id}: ", end="", flush=True)
        
        if gene_id in annotation_cache:
            print("cached")
            record = annotation_cache[gene_id]
            annotation_records.append(record)
            continue
        
        record = {
            "gene_id": gene_id,
            "log2FC": log2fc,
            "p_value": pvalue,
            "uniprot_id": None,
            "protein_name": "Unknown",
            "function": "Not available",
            "GO_biological_process": "",
            "GO_molecular_function": "",
            "GO_cellular_component": "",
            "InterPro_domains": "",
            "Pfam_domains": "",
            "CATH_domains": "",
            "subcellular_location": "",
            "protein_length": 0,
            "ensembl_description": "",
            "ensembl_chromosome": "",
            "alphafold_confidence": "N/A",
            "alphafold_version": "Unknown",
            "secondary_structure": "",
            "secondary_confidence": "",
            "rapid_fold_model": "",
            "rapid_fold_path": "",
            "psipred_states": "",
            "psipred_confidence": "",
            "tmhmm_segments": "",
            "tmhmm_topology": "",
            "tmhmm_plot": "",
            "tmhmm_is_transporter": False,
            "orthologous_groups": "",
            "place_location": ""
        }
        
        status_parts = []
        
        try:
            # ENSEMBL
            ens = ensembl_lookup(gene_id)
            if ens and isinstance(ens, dict):
                record["ensembl_description"] = ens.get("description", "")
                record["ensembl_chromosome"] = ens.get("seq_region_name", "")
                if not record["protein_name"] or record["protein_name"] == "Unknown":
                    record["protein_name"] = ens.get("display_name", "Unknown")
                status_parts.append("ENS")
            
            # UNIPROT
            acc = uniprot_search(gene_id, uniprot_cache)
            record["uniprot_id"] = acc
            print(f"UniProt → {acc}")
            
            if acc:
                status_parts.append(f"UP:{acc}")
                uni = uniprot_entry(acc)

                if uni and isinstance(uni, dict):
                    # =============================
                    # UniProt basic annotations
                    # =============================
                    try:
                        pname = (uni.get("proteinDescription", {})
                                .get("recommendedName", {})
                                .get("fullName", {})
                                .get("value"))
                        if pname:
                            record["protein_name"] = pname
                    except:
                        pass

                    for comment in uni.get("comments", []):
                        try:
                            if comment.get("commentType") == "FUNCTION":
                                texts = comment.get("texts", [])
                                if texts:
                                    record["function"] = texts[0].get("value", "")[:800]
                            elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                                locs = []
                                for loc in comment.get("subcellularLocations", []):
                                    location = loc.get("location", {})
                                    if location:
                                        locs.append(location.get("value", ""))
                                record["subcellular_location"] = "; ".join(locs)
                        except:
                            continue

                    try:
                        record["protein_length"] = uni.get("sequence", {}).get("length", 0)
                    except:
                        pass

                    # =============================
                    # GO terms and domains
                    # =============================
                    go_bp, go_mf, go_cc = [], [], []
                    interpro_ids, pfam_ids, cath_ids = [], [], []

                    for xref in uni.get("uniProtKBCrossReferences", []):
                        try:
                            db = xref.get("database")

                            if db == "GO":
                                props = xref.get("properties", [])
                                if props:
                                    go_term = props[0].get("value", "")
                                    go_id = xref.get("id", "")

                                    if "P:" in go_term:
                                        go_bp.append(f"{go_id} ({go_term.split('P:')[1].strip()})")
                                    elif "F:" in go_term:
                                        go_mf.append(f"{go_id} ({go_term.split('F:')[1].strip()})")
                                    elif "C:" in go_term:
                                        go_cc.append(f"{go_id} ({go_term.split('C:')[1].strip()})")

                            elif db == "InterPro":
                                interpro_ids.append(xref.get("id", ""))
                            elif db == "Pfam":
                                pfam_ids.append(xref.get("id", ""))
                            elif db == "Gene3D":
                                cath_ids.append(xref.get("id", ""))
                        except:
                            continue

                    record["GO_biological_process"] = "; ".join(go_bp[:10])
                    record["GO_molecular_function"] = "; ".join(go_mf[:10])
                    record["GO_cellular_component"] = "; ".join(go_cc[:10])
                    record["InterPro_domains"] = "; ".join(interpro_ids[:10])
                    record["Pfam_domains"] = "; ".join(pfam_ids[:10])
                    record["CATH_domains"] = "; ".join(cath_ids[:10])

                    # =============================
                    # FASTA download
                    # =============================
                    if download_file(
                        f"{UNIPROT_ENTRY}/{acc}.fasta",
                        f"{gene_dir}/protein_{gene_id}.fasta",
                        binary=False
                    ):
                        status_parts.append("FASTA")

                    protein_seq = ""
                    try:
                        with open(f"{gene_dir}/protein_{gene_id}.fasta", "r") as f_fa:
                            protein_seq = extract_sequence_from_fasta(f_fa.read())
                    except Exception:
                        protein_seq = ""

                    if protein_seq:
                        protein_seq = clean_protein_sequence(protein_seq)
                        sec_pred = {"states": "", "confidence": [], "formatted_lines": [], "method": "none"}
                        tmhmm = None
                        if not protein_seq or len(protein_seq) < 20:
                            status_parts.append("SEQ:too_short_or_invalid")
                        else:
                            sec_pred = predict_secondary_structure(
                                protein_seq,
                                acc,
                                gene_dir=gene_dir,
                                gene_id=gene_id,
                            )
                            if sec_pred.get("method") == "gor4":
                                status_parts.append("GOR4")
                        record["secondary_structure"] = sec_pred.get("states", "")
                        record["secondary_confidence"] = json.dumps(sec_pred.get("confidence", []))
                        record["psipred_states"] = sec_pred.get("states", "")
                        record["psipred_confidence"] = json.dumps(sec_pred.get("confidence", []))

                        # Save secondary structure to text file for downstream viewing
                        try:
                            with open(f"{gene_dir}/secondary_{gene_id}.txt", "w") as f_sec:
                                formatted_lines = sec_pred.get("formatted_lines") or []
                                if formatted_lines:
                                    f_sec.write("\n".join(formatted_lines))
                                else:
                                    f_sec.write(f"Sequence\n{protein_seq}\n")
                                    f_sec.write("|" * len(protein_seq) + "\n")
                                    f_sec.write(sec_pred.get("states", ""))
                        except Exception:
                            pass

                        if protein_seq and len(protein_seq) >= 20:
                            rapid = run_esmf_fold(protein_seq, f"{gene_dir}/fold_rapid_{gene_id}.pdb")
                            if rapid:
                                record["rapid_fold_model"] = rapid.get("model", "ESMFold")
                                record["rapid_fold_path"] = rapid.get("pdb_path", "")
                                status_parts.append("FOLD:rapid")

                            # Phobius submission (simple one-shot, saves text and PNG)
                            tm_txt = f"{gene_dir}/tmhmm_{gene_id}.txt"
                            tm_png = f"{gene_dir}/tmhmm_{gene_id}.png"
                            if os.path.exists(tm_txt) and os.path.exists(tm_png):
                                status_parts.append("PHOBIUS:cached")
                            else:
                                try:
                                    fasta_payload = f">{gene_id}\n{protein_seq}\n"
                                    ph_resp = requests.post(
                                        f"{PHOBIUS_BASE}/run",
                                        data={
                                            "sequence": fasta_payload,
                                            "email": PHOBIUS_EMAIL,
                                            "format": "grp",
                                            "toolId": "phobius",
                                            "stype": "protein",
                                            "database": "pfam-a",
                                        },
                                        timeout=30,
                                    )
                                    ph_resp.raise_for_status()
                                    job_id = ph_resp.text.strip()
                                    print(f"    Phobius job submitted: {job_id}")

                                    st_resp = requests.get(f"{PHOBIUS_BASE}/status/{job_id}", timeout=30)
                                    st_resp.raise_for_status()
                                    status = st_resp.text.strip()
                                    print(f"    Phobius status: {status}; waiting 4s before fetch")
                                    time.sleep(4)

                                    out_txt_url = f"{PHOBIUS_BASE}/result/{job_id}/out"
                                    out_png_url = f"{PHOBIUS_BASE}/result/{job_id}/visual-png"

                                    r_txt = requests.get(out_txt_url, timeout=60)
                                    r_txt.raise_for_status()
                                    with open(tm_txt, "wb") as f_txt:
                                        f_txt.write(r_txt.content)

                                    r_png = requests.get(out_png_url, timeout=60)
                                    if r_png.status_code == 200 and r_png.content:
                                        with open(tm_png, "wb") as f_png:
                                            f_png.write(r_png.content)
                                    status_parts.append("PHOBIUS")
                                except Exception as exc:
                                    print(f"    Phobius failed: {exc}")

                        # PLACE 2.0-style heuristic: prefer UniProt subcellular location, otherwise infer from TM spans
                        if record.get("subcellular_location"):
                            record["place_location"] = record["subcellular_location"]
                        elif tmhmm and tmhmm.get("is_transporter"):
                            record["place_location"] = "Membrane (PLACE 2.0-style heuristic via TM spans)"
                        else:
                            record["place_location"] = "Cytosolic/peripheral (heuristic)"

                    # =============================
                    # AlphaFold
                    # =============================
                    af_result = download_alphafold_pdb(acc,f"{gene_dir}/structure_{gene_id}.pdb")

                    if af_result:
                        af_info = get_alphafold_info(acc)
                        if af_info:
                            record["alphafold_confidence"] = af_info.get("confidence", "N/A")
                        record["alphafold_version"] = af_result["model_version"]
                        status_parts.append("AF")


                    # =============================
                    # KEGG
                    # =============================
                    kegg_genes = uniprot_to_kegg_genes(acc)

                    for kegg_gene in kegg_genes:
                        kos = kegg_gene_to_kos(kegg_gene)
                        pathways = kegg_gene_to_pathways(kegg_gene)

                        for ko in kos:
                            for pathway in pathways:
                                kegg_records.append({
                                    "gene_id": gene_id,
                                    "uniprot_id": acc,
                                    "KEGG_gene_id": kegg_gene,
                                    "KO": ko,
                                    "KEGG_pathway_id": pathway
                                })

                    if kegg_genes:
                        status_parts.append(f"KEGG:{len(kegg_genes)}")

                    # =============================
                    # EggNOG
                    # =============================
                    eggnog_ids = get_eggnog_ids_from_uniprot_json(uni)

                    # Fallback: direct EggNOG lookup by UniProt if cross-references are missing
                    if not eggnog_ids:
                        fallback_eggnog = get_eggnog_from_uniprot(acc)
                        extracted = []
                        if isinstance(fallback_eggnog, dict):
                            pools = []
                            for key in ["data", "results", "hits"]:
                                val = fallback_eggnog.get(key)
                                if isinstance(val, list):
                                    pools.extend(val)
                            if not pools:
                                pools = [fallback_eggnog]
                            for entry in pools:
                                if not isinstance(entry, dict):
                                    continue
                                ogid = entry.get("og_id") or entry.get("og") or entry.get("nog") or entry.get("orthologous_group")
                                if ogid:
                                    extracted.append(ogid)
                        elif isinstance(fallback_eggnog, list):
                            for entry in fallback_eggnog:
                                if isinstance(entry, dict):
                                    ogid = entry.get("og_id") or entry.get("og") or entry.get("nog")
                                    if ogid:
                                        extracted.append(ogid)
                        if extracted:
                            # preserve order while deduplicating
                            seen = set()
                            eggnog_ids = []
                            for ogid in extracted:
                                if ogid not in seen:
                                    seen.add(ogid)
                                    eggnog_ids.append(ogid)

                    if eggnog_ids:
                        record["orthologous_groups"] = ";".join([f"{e} (EggNOG/COG)" for e in eggnog_ids])
                    eggnog_count = 0

                    for nog_id in eggnog_ids[:3]:
                        go_data = get_eggnog_annotation(nog_id, "go_terms")
                        domain_data = get_eggnog_annotation(nog_id, "domains")
                        ortho_data = get_eggnog_orthologs(nog_id)

                        if go_data and "go_terms" in go_data:
                            for category, go_list in go_data["go_terms"].items():
                                for go_entry in go_list:
                                    if isinstance(go_entry, (list, tuple)) and len(go_entry) >= 2:
                                        eggnog_go_records.append({
                                            "gene_id": gene_id,
                                            "uniprot_id": acc,
                                            "eggnog_id": nog_id,
                                            "GO_category": category,
                                            "GO_ID": go_entry[0],
                                            "GO_name": go_entry[1]
                                        })
                                        eggnog_count += 1

                        if domain_data and "domains" in domain_data:
                            for domain_type, domain_list in domain_data["domains"].items():
                                for domain_entry in domain_list:
                                    if isinstance(domain_entry, (list, tuple)) and len(domain_entry) >= 3:
                                        eggnog_domain_records.append({
                                            "gene_id": gene_id,
                                            "uniprot_id": acc,
                                            "eggnog_id": nog_id,
                                            "domain_type": domain_type,
                                            "domain_name": domain_entry[0]
                                        })
                                        eggnog_count += 1

                        if ortho_data and isinstance(ortho_data, dict) and ortho_data.get("data"):
                            ortho_list = []
                            for group in ortho_data.get("data", []):
                                try:
                                    ogid = group.get("og_id")
                                    tax = group.get("tax_id")
                                    name = group.get("name") or ""
                                    if ogid:
                                        ortho_list.append(f"{ogid} ({name or tax} via EggNOG/COG)")
                                except Exception:
                                    continue
                            if ortho_list:
                                record["orthologous_groups"] = ";".join(sorted(set(ortho_list)))

                    if eggnog_count > 0:
                        status_parts.append(f"EGNOG:{eggnog_count}")

                    # =============================
                    # InterPro
                    # =============================
                    if interpro_ids:
                        interpro_count = 0
                        for ipr in interpro_ids[:3]:
                            r_ipr = safe_request(
                                f"{INTERPRO_API}/{ipr}",
                                params={"page_size": 1},
                                timeout=30
                            )
                            if r_ipr:
                                meta = safe_json(r_ipr)
                                if meta:
                                    interpro_records.append({
                                        "gene_id": gene_id,
                                        "uniprot_id": acc,
                                        "interpro_id": ipr,
                                        "domain_name": meta.get("metadata", {}).get("name", {}).get("name")
                                    })
                                    interpro_count += 1

                        if interpro_count > 0:
                            status_parts.append(f"IPR:{interpro_count}")

            
            # STRING Network
            svg_url = f"{STRING_API}/svg/network?identifiers={gene_id}&species={STRING_SPECIES}"
            r_svg = safe_request(svg_url, timeout=30)
            if r_svg and r_svg.content:
                with open(f"{gene_dir}/string_{gene_id}.svg", "wb") as f:
                    f.write(r_svg.content)
            
            tsv_text = get_string_tsv(gene_id)
            if tsv_text:
                lines = tsv_text.strip().split("\n")[1:]
                edges = []
                for l in lines:
                    c = l.split("\t")
                    if len(c) >= 6:
                        edges.append({
                            "source": c[2],
                            "target": c[3],
                            "combined_score": float(c[5])
                        })
                if edges:
                    pd.DataFrame(edges).to_csv(f"{gene_dir}/string_edges_{gene_id}.csv", index=False)
                    status_parts.append(f"STR:{len(edges)}")
            
            annotation_cache[gene_id] = record
            print(" | ".join(status_parts) if status_parts else "minimal")
            
        except Exception as e:
            print(f"error: {str(e)[:30]}")
        
        annotation_records.append(record)
        
        # Save cache every 10 genes
        if idx % 10 == 0:
            save_cache("annotations.json", annotation_cache)
            save_cache("uniprot_search.json", uniprot_cache)
            save_cache("eggnog.json", eggnog_cache)
            save_cache("kegg.json", kegg_cache)
            save_cache("alphafold.json", alphafold_cache)
            save_cache("uniprot_entry.json", uniprot_entry_cache)

    # Save all data
    ann_df = pd.DataFrame(annotation_records)
    ann_df.to_csv(f"{OUTPUT_DIR}/annotations.csv", index=False)
    
    kegg_df = pd.DataFrame(kegg_records) if kegg_records else pd.DataFrame()
    if not kegg_df.empty:
        kegg_df.to_csv(f"{OUTPUT_DIR}/kegg_pathways.csv", index=False)
    else:
        pd.DataFrame(columns=[
    "gene_id",
    "uniprot_id",
    "KO",
    "KEGG_pathway_id",
    "KEGG_pathway_name",
    "KEGG_pathway_class",
    "KEGG_description"
]).to_csv(f"{OUTPUT_DIR}/kegg_pathways.csv", index=False)

    eggnog_go_df = pd.DataFrame(eggnog_go_records) if eggnog_go_records else pd.DataFrame()
    if not eggnog_go_df.empty:
        eggnog_go_df.to_csv(f"{OUTPUT_DIR}/eggnog_go_terms.csv", index=False)
    else:
        pd.DataFrame(columns=[
    "gene_id",
    "uniprot_id",
    "eggnog_id",
    "GO_category",
    "GO_ID",
    "GO_name",
    "evidence",
    "gene_count",
    "ratio",
    "coverage_percent"
]).to_csv(f"{OUTPUT_DIR}/eggnog_go_terms.csv", index=False)
    
    eggnog_domains_df = pd.DataFrame(eggnog_domain_records) if eggnog_domain_records else pd.DataFrame()
    if not eggnog_domains_df.empty:
        eggnog_domains_df.to_csv(f"{OUTPUT_DIR}/eggnog_domains.csv", index=False)
    else:
        pd.DataFrame(columns=[
    "gene_id",
    "uniprot_id",
    "eggnog_id",
    "domain_type",
    "domain_name",
    "frequency",
    "gene_count",
    "coverage_percent"
]).to_csv(f"{OUTPUT_DIR}/eggnog_domains.csv", index=False)

    
    interpro_df = pd.DataFrame(interpro_records) if interpro_records else pd.DataFrame()
    if not interpro_df.empty:
        interpro_df.to_csv(f"{OUTPUT_DIR}/interpro_domains.csv", index=False)
    else:
        pd.DataFrame(columns=[
    "gene_id",
    "uniprot_id",
    "interpro_id",
    "domain_name",
    "domain_short_name",
    "domain_type",
    "CATH_domains"
]).to_csv(f"{OUTPUT_DIR}/interpro_domains.csv", index=False)

    
    save_cache("annotations.json", annotation_cache)
    save_cache("uniprot_search.json", uniprot_cache)
    save_cache("eggnog.json", eggnog_cache)
    save_cache("kegg.json", kegg_cache)
    save_cache("alphafold.json", alphafold_cache)
    mark_step_complete("unified_annotation")

    print(f"\n  ✓ Annotated {len(annotation_records)} genes")
    print(f"  ✓ KEGG pathways: {len(kegg_records)}")
    print(f"  ✓ EggNOG GO terms: {len(eggnog_go_records)}")
    print(f"  ✓ EggNOG domains: {len(eggnog_domain_records)}")
    print(f"  ✓ InterPro annotations: {len(interpro_records)}")

# Load saved data files
def safe_read_csv(path):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        pass
    return pd.DataFrame()

kegg_df = safe_read_csv(f"{OUTPUT_DIR}/kegg_pathways.csv")
eggnog_go_df = safe_read_csv(f"{OUTPUT_DIR}/eggnog_go_terms.csv")
eggnog_domains_df = safe_read_csv(f"{OUTPUT_DIR}/eggnog_domains.csv")
interpro_df = safe_read_csv(f"{OUTPUT_DIR}/interpro_domains.csv")
if "gene_id" not in ann_df.columns:
    ann_df["gene_id"] = []
# Ensure new optional columns exist for downstream consumers
for optional_col in [
    "secondary_structure",
    "secondary_confidence",
    "rapid_fold_model",
    "rapid_fold_path",
    "psipred_states",
    "psipred_confidence",
    "tmhmm_segments",
    "tmhmm_topology",
    "tmhmm_plot",
    "tmhmm_is_transporter",
    "orthologous_groups",
]:
    if optional_col not in ann_df.columns:
        ann_df[optional_col] = ""
ann_index = ann_df.set_index("gene_id", drop=False)

# =============================
# SUMMARY STATISTICS
# =============================

# Top KEGG pathways
top_kegg = (
    kegg_df["KEGG_pathway_id"]
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={"index": "KEGG_pathway_id", "KEGG_pathway_id": "Gene_count"})
) if not kegg_df.empty else pd.DataFrame()

# Top GO Biological Processes
top_go_bp = (
    eggnog_go_df[eggnog_go_df["GO_category"] == "BP"]["GO_name"]
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={"index": "GO_Biological_Process", "GO_name": "Gene_count"})
) if not eggnog_go_df.empty else pd.DataFrame()

# Top STRING hub proteins (degree based)
string_edges = []

# Collect edges from upregulated and downregulated networks
for path in [
    f"{UP_NETWORK}/upregulated_edges.csv",
    f"{DOWN_NETWORK}/downregulated_edges.csv"
]:
    if os.path.exists(path):
        df_edges = pd.read_csv(path)
        if not df_edges.empty:
            string_edges.append(df_edges[["source", "target"]])

if string_edges:
    all_edges = pd.concat(string_edges)
    hubs = (
        pd.concat([
            all_edges["source"],
            all_edges["target"]
        ])
        .value_counts()
        .head(10)
        .reset_index()
        .rename(columns={"index": "Protein", 0: "Interaction_count"})
    )
else:
    hubs = pd.DataFrame()



# ==========================================================
# STRING SUBNETWORKS
# ==========================================================
print("\n[STEP 3/4] Building STRING subnetworks...")

if is_step_complete_with_files("string_networks", [
    f"{UP_NETWORK}/upregulated_edges.csv",
    f"{DOWN_NETWORK}/downregulated_edges.csv"
]):
    print("  ✓ Using cached networks")
else:
    print("  Building protein-protein interaction networks...")
    
    def build_subnetwork(gene_df, network_folder, filename_prefix):
        gene_list = list(gene_df["gene_id"])
        genes_string = "%0d".join(gene_list)
        
        print(f"    {filename_prefix}: ", end="", flush=True)
        
        
        network_url = f"{STRING_API}/tsv/network?identifiers={genes_string}&species={STRING_SPECIES}&required_score=400"
        r = safe_request(network_url, timeout=60)
        edges = []
        if r and r.text.strip():
            lines = r.text.strip().split("\n")[1:]
            for line in lines:
                cols = line.split("\t")
                if len(cols) >= 6:
                    edges.append({"source": cols[2], "target": cols[3], "combined_score": float(cols[5])})
            if edges:
                pd.DataFrame(edges).to_csv(f"{network_folder}/{filename_prefix}_edges.csv", index=False)
                print(f"{len(edges)} edges", end=" | ")
            else:
                # Create empty file
                pd.DataFrame(columns=["source", "target", "combined_score"]).to_csv(
                    f"{network_folder}/{filename_prefix}_edges.csv", index=False)
                print("no edges", end=" | ")
        else:
            # Create empty file
            pd.DataFrame(columns=["source", "target", "combined_score"]).to_csv(
                f"{network_folder}/{filename_prefix}_edges.csv", index=False)
            print("no data", end=" | ")
        
        
        svg_url = f"{STRING_API}/svg/network?identifiers={genes_string}&species={STRING_SPECIES}"
        svg_resp = safe_request(svg_url, timeout=60)
        if svg_resp and svg_resp.content:
            with open(f"{network_folder}/{filename_prefix}_network.svg", "wb") as f:
                f.write(svg_resp.content)
            print("svg saved")
        else:
            print("no svg")

    build_subnetwork(upregulated, UP_NETWORK, "upregulated")
    build_subnetwork(downregulated, DOWN_NETWORK, "downregulated")
    mark_step_complete("string_networks")

# ==========================================================
# VOLCANO PLOT
# ==========================================================
print("\n[STEP 4/4] Creating volcano plot...")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["log2FC"], y=df["neg_log10_p"],
    mode="markers", name="Non-significant",
    marker=dict(color="black", size=4, opacity=0.5),
    hoverinfo="skip",
    customdata=np.column_stack((df["gene_id"].values, ["non-significant"] * len(df)))
))

def add_genes(data, color, label):
    if data.empty:
        return
    merged = data.merge(ann_df, on="gene_id", how="left")

    hover_text = []
    for _, r in merged.iterrows():
        func_preview = str(r.get('function', 'Unknown'))[:100]
        # Fixed-width container keeps hover boxes visually consistent and prevents spillover
        text = (
            "<div style='width:220px; white-space:normal; overflow-wrap:break-word; word-break:break-word;'>"
            f"<b>Gene:</b> {r['gene_id']}<br>"
            f"<b>Protein:</b> {r.get('protein_name', 'Unknown')}<br>"
            f"<b>log2FC:</b> {r['log2FC_x']:.3f}<br>"
            f"<b>Raw p-value:</b> {r['p_value_x']:.2e}<br>"
            f"<b>FDR-adjusted p-value:</b> {r.get('adj_p_value', r['p_value_x']):.2e}<br>"
            f"<b>Function:</b> {func_preview}..."
            "</div>"
        )
        hover_text.append(text)

    fig.add_trace(go.Scatter(
        x=merged["log2FC_x"], y=merged["neg_log10_p"],
        mode="markers", name=label,
        marker=dict(color=color, size=8, opacity=0.8, line=dict(width=0.5, color='white')),
        customdata=np.column_stack((merged["gene_id"].values, [label.lower()] * len(merged))),
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover_text,
        hoverlabel=dict(bgcolor=color, font=dict(color="white", size=15), bordercolor="#111")
    ))

add_genes(upregulated, "#00CC66", "Upregulated")
add_genes(downregulated, "#FF4444", "Downregulated")

fig.add_vline(x=LOG2FC_THRESHOLD, line_dash="dash", line_color="black", opacity=0.5)
fig.add_vline(x=-LOG2FC_THRESHOLD, line_dash="dash", line_color="black", opacity=0.5)
fig.add_hline(y=-np.log10(PVALUE_THRESHOLD), line_dash="dash", line_color="black", opacity=0.5)

fig.update_layout(
    title={"text": "Interactive Volcano Plot - Rice Gene Expression", "x": 0.5, "xanchor": "center"},
    xaxis_title="log2 Fold Change",
    yaxis_title=f"-log10 ({'FDR-adjusted' if USE_ADJUSTED_PVALUE else 'raw'} p-value)",
    template="plotly_white",
    height=700,
    hovermode="closest",
    hoverlabel=dict(
        font=dict(size=15, color="white"),
        bgcolor="rgba(31, 41, 55, 0.9)",
        bordercolor="#E5E7EB",
        align="left"
    ),
    font=dict(size=12)
)

# ==========================================================
# DASH WEB SERVER
# ==========================================================
if RUN_DASH_SERVER:
    print("\nStarting web server...")
else:
    # Dash server disabled; pipeline-only mode.
    pass

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Transcriptomics Pipeline Server"
app.config.suppress_callback_exceptions = True

app.layout = dbc.Container([
    html.H1("Rice Transcriptomics Analysis Pipeline", className="text-center my-4",
            style={"color": "#2C3E50"}),
    
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(len(df), className="text-center mb-0"),
            html.P("Total Genes", className="text-center text-muted small mb-0")
        ]), color="light"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(len(upregulated), className="text-center mb-0", style={"color": "#00CC66"}),
            html.P("Upregulated", className="text-center text-muted small mb-0")
        ]), color="light"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(len(downregulated), className="text-center mb-0", style={"color": "#FF4444"}),
            html.P("Downregulated", className="text-center text-muted small mb-0")
        ]), color="light"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(len(ann_df), className="text-center mb-0", style={"color": "#3498DB"}),
            html.P("Annotated", className="text-center text-muted small mb-0")
        ]), color="light"), width=3)
    ], className="mb-4"),

    dbc.Card([
    dbc.CardHeader(html.H5("Biological Summary Overview")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H6("Top KEGG Pathways"),
                dbc.Table.from_dataframe(
                    top_kegg,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm"
                ) if not top_kegg.empty else html.P("No KEGG data available")
            ], width=6),

            dbc.Col([
                html.H6("Top GO Biological Processes"),
                dbc.Table.from_dataframe(
                    top_go_bp,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm"
                ) if not top_go_bp.empty else html.P("No GO BP data available")
            ], width=6),
        ])
    ])
], className="mb-4"),
        dbc.Row([
    dbc.Col([
        html.H6("Top STRING Hub Proteins"),
        dbc.Table.from_dataframe(
            hubs,
            striped=True,
            bordered=True,
            hover=True,
            size="sm"
        ) if not hubs.empty else html.P("No STRING hub data available")
    ], width=12)
], className="mt-3"),



    dbc.Card([
        dbc.CardHeader(html.H5("Interactive Volcano Plot", className="mb-0")),
        dbc.CardBody([
            html.P("Click on upregulated (green) or downregulated (red) genes to view details.", 
                   className="text-muted small mb-3"),
            dcc.Graph(id="volcano", figure=fig, config={"displaylogo": False})
        ])
    ], className="mb-4"),
    # STRING UPregulated network
            html.Hr(),
            html.H5("STRING Upregulated Network Visualization", className="mb-2"),
            html.Div([
                html.Img(
                    src=f"data:image/svg+xml;base64,{get_base64_image(up_network_img)}",
                    style={"width": "100%", "maxWidth": "800px", "display": "block", "margin": "0 auto"}
                ) if os.path.exists(up_network_img) else html.P("Network visualization not available", className="text-center text-muted",
                                             style={"padding": "50px"})
            ]),
            #STRING Downregulated network
            html.Hr(),
            html.H5("STRING Downregulated Network Visualization", className="mb-2"),
            html.Div([
                html.Img(
                    src=f"data:image/svg+xml;base64,{get_base64_image(down_network_img)}",
                    style={"width": "100%", "maxWidth": "800px", "display": "block", "margin": "0 auto"}
                ) if os.path.exists(down_network_img) else html.P("Network visualization not available", className="text-center text-muted",
                                             style={"padding": "50px"})
            ]),


    html.Div(id="gene-panel")
], fluid=True, style={"backgroundColor": "#F8F9FA", "minHeight": "100vh"})

@app.callback(Output("gene-panel", "children"), Input("volcano", "clickData"))
def show_gene(clickData):
    if not clickData:
        return dbc.Alert("Click on upregulated (green) or downregulated (red) genes to view details.",
                        color="info", className="text-center")
    
    try:
        gene_id = clickData["points"][0]["customdata"][0]
        if gene_id in gene_panel_cache:
            return gene_panel_cache[gene_id]

        point_type = clickData["points"][0]["customdata"][1]
        
        if point_type == "non-significant":
            return dbc.Alert(f"Gene {gene_id} is not significantly differentially expressed. "
                           "Click on upregulated or downregulated genes to view annotations.", 
                           color="warning", className="text-center")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

    if gene_id not in ann_index.index:
        return dbc.Alert(f"No annotation data for {gene_id}", color="warning")

    row = ann_index.loc[gene_id]

    gene_dir = f"{UP_GENES_DIR}/{gene_id}" if row["log2FC"] > 0 else f"{DOWN_GENES_DIR}/{gene_id}"

    # Load files
    fasta_path = f"{gene_dir}/protein_{gene_id}.fasta"
    fasta_text = "FASTA not available"
    if os.path.exists(fasta_path):
        try:
            with open(fasta_path) as f:
                fasta_text = f.read()
        except:
            pass

    pdb_path = f"{gene_dir}/structure_{gene_id}.pdb"
    has_structure = os.path.exists(pdb_path)
    pdb_content = ""
    if has_structure:
        try:
            with open(pdb_path, 'r') as f:
                pdb_content = f.read()
        except:
            has_structure = False

    svg_path = f"{gene_dir}/string_{gene_id}.svg"
    has_network = os.path.exists(svg_path)
    
    edges_path = f"{gene_dir}/string_edges_{gene_id}.csv"
    has_edges = os.path.exists(edges_path)

    # Get gene-specific data
    kegg_data = []
    if not kegg_df.empty and "gene_id" in kegg_df.columns:
        kegg_data = kegg_df[kegg_df["gene_id"] == gene_id].to_dict("records")
    
    eggnog_go_data = []
    if not eggnog_go_df.empty and "gene_id" in eggnog_go_df.columns:
        eggnog_go_data = eggnog_go_df[eggnog_go_df["gene_id"] == gene_id].to_dict("records")
    
    eggnog_domain_data = []
    if not eggnog_domains_df.empty and "gene_id" in eggnog_domains_df.columns:
        eggnog_domain_data = eggnog_domains_df[eggnog_domains_df["gene_id"] == gene_id].to_dict("records")
    
    interpro_data = []
    if not interpro_df.empty and "gene_id" in interpro_df.columns:
        interpro_data = interpro_df[interpro_df["gene_id"] == gene_id].to_dict('records')

    # Get Cytoscape network
    cyto_nodes, cyto_edges = get_string_network_cytoscape(gene_id)
    
    reg = "Upregulated" if row["log2FC"] > 0 else "Downregulated"
    reg_color = "success" if row["log2FC"] > 0 else "danger"
    
    # AlphaFold info
    af_confidence = row.get("alphafold_confidence", "N/A")
    af_version = row.get("alphafold_version", "Unknown")
    af_link = f"https://alphafold.ebi.ac.uk/entry/{row['uniprot_id']}" if row["uniprot_id"] else None

    card = dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([html.H3([gene_id], className="mb-0")], width=8),
                dbc.Col([
                    dbc.Badge(reg, color=reg_color, className="float-end", style={"fontSize": "1rem"})
                ], width=4)
            ])
        ], style={"backgroundColor": "#2C3E50", "color": "white"}),

        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Expression Profile", className="mb-3"),
                    html.Table([
                        html.Tr([
                            html.Td("log2 Fold Change:", style={"fontWeight": "bold", "width": "180px"}),
                            html.Td(f"{row['log2FC']:.4f}",
                                style={"color": "#00CC66" if row['log2FC'] > 0 else "#FF4444"})
                        ]),
                        html.Tr([
                            html.Td("Raw p-value:", style={"fontWeight": "bold"}),
                            html.Td(f"{row.get('p_value', 'N/A'):.2e}" if pd.notna(row.get('p_value')) else "N/A")
                        ]),
                        html.Tr([
                            html.Td("Adjusted p-value (FDR):", style={"fontWeight": "bold"}),
                            html.Td(f"{row.get('adj_p_value', row.get('p_value')):.2e}")
                        ]),
                        html.Tr([
                            html.Td("-log10(FDR):", style={"fontWeight": "bold"}),
                            html.Td(f"{-np.log10(row.get('adj_p_value', row.get('p_value'))):.3f}")
                        ])
                    ], className="table table-sm table-borderless")
                ], width=6),

                dbc.Col([
                    html.H5("Protein Information", className="mb-3"),
                    html.Table([
                        html.Tr([
                            html.Td("Name:", style={"fontWeight": "bold", "width": "100px"}),
                            html.Td(row["protein_name"])
                        ]),
                        html.Tr([
                            html.Td("UniProt:", style={"fontWeight": "bold"}),
                            html.Td(row["uniprot_id"] or "N/A")
                        ]),
                        html.Tr([
                            html.Td("Length:", style={"fontWeight": "bold"}),
                            html.Td(f"{row['protein_length']} aa" if row["protein_length"] > 0 else "Unknown")
                        ])
                    ], className="table table-sm table-borderless")
                ], width=6)
            ], className="mb-4"),

            html.Hr(),

            html.H5("Biological Function", className="mb-2"),
            html.P(row["function"] if row["function"] != "Not available" else "Function information not available",
                className="small"),

            html.Hr(),

            html.H5("Gene Ontology", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("Biological Process", className="mb-2"),
                    html.Div(row["GO_biological_process"] or "N/A", className="small",
                        style={"maxHeight": "150px", "overflowY": "auto", "padding": "10px",
                               "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
                ], width=4),

                dbc.Col([
                    html.H6("Molecular Function", className="mb-2"),
                    html.Div(row["GO_molecular_function"] or "N/A", className="small",
                        style={"maxHeight": "150px", "overflowY": "auto", "padding": "10px",
                               "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
                ], width=4),

                dbc.Col([
                    html.H6("Cellular Component", className="mb-2"),
                    html.Div(row["GO_cellular_component"] or "N/A", className="small",
                        style={"maxHeight": "150px", "overflowY": "auto", "padding": "10px",
                               "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
                ], width=4)
            ], className="mb-4"),

            html.Hr(),

            html.H5("KEGG Pathways", className="mb-3"),
            html.Div([
                dbc.Table.from_dataframe(
                    pd.DataFrame(kegg_data)[[
                        "KEGG_gene_id", "KO", "KEGG_pathway_id"
                    ]] if kegg_data else pd.DataFrame(),
                    striped=True, bordered=True, hover=True, size="sm"
                ) if kegg_data else html.P("No KEGG pathway data available", className="text-muted")
            ], style={"maxHeight": "300px", "overflowY": "auto"}),

            html.Hr(),

            html.H5("EggNOG GO Annotations", className="mb-3"),
            html.Div([
                dbc.Table.from_dataframe(
                    pd.DataFrame(eggnog_go_data)[[
                        "eggnog_id","GO_category", "GO_ID", "GO_name"
                    ]] if eggnog_go_data else pd.DataFrame(),
                    striped=True, bordered=True, hover=True, size="sm"
                ) if eggnog_go_data else html.P("No EggNOG GO data available", className="text-muted")
            ], style={"maxHeight": "300px", "overflowY": "auto"}),

            html.Hr(),

            html.H5("EggNOG Domain Annotations", className="mb-3"),
            html.Div([
                dbc.Table.from_dataframe(
                    pd.DataFrame(eggnog_domain_data)[[
                "eggnog_id","domain_name","domain_type"
                                    ]] if eggnog_domain_data else pd.DataFrame(),
                    striped=True, bordered=True, hover=True, size="sm"
                ) if eggnog_domain_data else html.P("No EggNOG domain data available", className="text-muted")
            ], style={"maxHeight": "300px", "overflowY": "auto"}),

            html.Hr(),

            html.H5("Protein Domains", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("InterPro"),
                    html.Div(row["InterPro_domains"] or "None", className="small")
                ], width=4),
                dbc.Col([
                    html.H6("Pfam"),
                    html.Div(row["Pfam_domains"] or "None", className="small")
                ], width=4),
                dbc.Col([
                    html.H6("CATH"),
                    html.Div(row["CATH_domains"] or "None", className="small")
                ], width=4)
            ], className="mb-4"),

            html.Hr(),

            html.H5("InterPro Detailed Annotations", className="mb-3"),
            html.Div([
                dbc.Table.from_dataframe(
                    pd.DataFrame(interpro_data)[[
                        "interpro_id", "domain_name"
                    ]] if interpro_data else pd.DataFrame(),
                    striped=True, bordered=True, hover=True, size="sm"
                ) if interpro_data else html.P("No InterPro annotations available", className="text-muted")
            ]),

            html.Hr(),
            
            html.H5("Protein Sequence (FASTA)", className="mb-2"),
            html.Pre(fasta_text, style={
                "maxHeight": "200px", "overflowY": "auto", "fontSize": "12px",
                "backgroundColor": "#1C344B", "color": "#FFFFFF", "padding": "15px",
                "borderRadius": "5px", "fontFamily": "monospace"
            }),
            
            html.Hr(),
            
            html.Div([
                html.H5("AlphaFold Predicted 3D Structure", className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.B("Model Version: "), af_version,
                            html.Br(),
                            html.B("Global Confidence Score: "), str(af_confidence),
                            html.Br(),
                            html.A("View on AlphaFold Database", href=af_link, target="_blank", 
                                  className="btn btn-sm btn-primary mt-2") if af_link else ""
                        ], className="small mb-3")
                    ])
                ]),
                html.Div([
                    html.Iframe(
                        srcDoc=f'''
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <script src="https://cdn.jsdelivr.net/npm/ngl@2.0.0-dev.36/dist/ngl.js"></script>
                            <style>
                                body {{ margin: 0; padding: 0; }}
                                #viewport {{ width: 100%; height: 400px; }}
                            </style>
                        </head>
                        <body>
                            <div id="viewport"></div>
                            <script>
                                var stage = new NGL.Stage("viewport", {{backgroundColor: "white"}});
                                var pdbData = `{pdb_content}`;
                                var blob = new Blob([pdbData], {{type: 'text/plain'}});
                                stage.loadFile(blob, {{ext: "pdb"}}).then(function(component) {{
                                    component.addRepresentation("cartoon", {{colorScheme: "bfactor"}});
                                    component.addRepresentation("ball+stick", {{sele: "hetero"}});
                                    component.autoView();
                                }});
                            </script>
                        </body>
                        </html>
                        ''' if has_structure and pdb_content else '''
                        <div style="padding: 50px; text-align: center; color: #999;">
                            3D structure not available for this gene
                        </div>
                        ''',
                        style={"width": "100%", "height": "420px", "border": "2px solid #ddd", "borderRadius": "5px"}
                    ) if has_structure else html.P("3D structure not available", className="text-center text-muted", 
                                                   style={"padding": "50px"})
                ])
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Interactive STRING Network (Cytoscape)", className="mb-2"),
                    html.P(
                        "Explore protein-protein interactions. Blue node = query protein, colored nodes = interaction clusters.",
                        className="small text-muted"
                    ),
                    html.Div([
                        cyto.Cytoscape(
                            id=f'cyto-{gene_id}',
                            layout={'name': 'cose', 'padding': 50, 'animate': False},
                            style={'width': '100%', 'height': '500px', 'border': '2px solid #ddd', 'borderRadius': '5px'},
                            elements=cyto_nodes + cyto_edges,
                            minZoom=0.1,
                            maxZoom=2,
                            zoomingEnabled=True,
                            userZoomingEnabled=True,
                            wheelSensitivity=0.1,
                            stylesheet=[
                                {
                                    'selector': 'node',
                                    'style': {
                                        'label': 'data(label)',
                                        'font-size': '8px',
                                        'width': 25,
                                        'height': 25,
                                        'background-color': 'data(cluster_color)'
                                    }
                                },
                                {
                                    'selector': 'edge',
                                    'style': {
                                        'width': 2,
                                        'line-color': '#ccc'
                                    }
                                }
                            ]
                        ) if cyto_nodes else html.P(
                            "Interactive network not available",
                            className="text-center text-muted",
                            style={"padding": "50px"}
                        )
                    ])
                ], xs=12, sm=12, md=6, lg=6),

                dbc.Col([
                    html.H5("STRING Network Visualization", className="mb-2"),
                    html.Div([
                        html.Img(
                            src=f"data:image/svg+xml;base64,{get_base64_image(svg_path)}",
                            style={"width": "100%", "maxWidth": "800px", "display": "block", "margin": "0 auto"}
                        ) if has_network else html.P(
                            "Network visualization not available",
                            className="text-center text-muted",
                            style={"padding": "50px"}
                        )
                    ])
                ], xs=12, sm=12, md=6, lg=6)
            ], className="g-4 align-items-start"),

            html.Hr() if has_edges else html.Div(),
            
            html.H5("STRING Network Interactions", className="mb-2") if has_edges else html.Div(),
            html.Div([
                dbc.Table.from_dataframe(
                    pd.read_csv(edges_path),
                    striped=True, bordered=True, hover=True, size="sm"
                ) if has_edges else html.Div()
            ], style={"maxHeight": "300px", "overflowY": "auto"}) if has_edges else html.Div()
        ]),
        
    ], className="shadow-sm")
    gene_panel_cache[gene_id] = card
    return card



#### End Timer###
end_time = time.time()
total_time = end_time - start_time

print(f"Total time taken to run the analysis >>> {int(total_time // 60)} minutes {int(total_time%60)} sec")
############



if __name__ == "__main__" and RUN_DASH_SERVER:
    print("\n" + "=" * 60)
    print(" SERVER READY!")
    print("=" * 60 + "\n")
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=8055)
elif __name__ == "__main__":
    # Dash UI skipped. Set RUN_DASH_SERVER=1 to launch the dashboard.
    pass