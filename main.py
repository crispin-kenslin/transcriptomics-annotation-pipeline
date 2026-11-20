import os
import sys
import time
import json
import csv
import base64
import hashlib
import subprocess
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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
PLANT_SPECIES_KEY = os.getenv("PLANT_SPECIES_KEY", "rice")
PLANT_SPECIES_LABEL = os.getenv("PLANT_SPECIES_LABEL", "Rice")
PLANT_SPECIES_SHORT_NAME = os.getenv("PLANT_SPECIES_SHORT_NAME", PLANT_SPECIES_LABEL)
SPECIES_SCIENTIFIC_NAME = os.getenv(
    "SPECIES_SCIENTIFIC_NAME", PLANT_SPECIES_LABEL
)
UNIPROT_TAX_ID = os.getenv("UNIPROT_TAX_ID", "39947")

               
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/files"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
STRING_API = "https://string-db.org/api"
STRING_SPECIES = int(os.getenv("STRING_SPECIES", "39947"))
REQUEST_CLIENT_IP = os.getenv("REQUEST_CLIENT_IP", "unknown")
REQUEST_CLIENT_API_KEY = os.getenv("REQUEST_CLIENT_API_KEY", "")
TMHMM_CACHE_DIR = os.getenv("TMHMM_CACHE_DIR", f"{CACHE_DIR}/tmhmm")
TMHMM_PACKAGE_ROOT = Path(__file__).resolve().parent
TMHMM_MODEL = Path(os.getenv("TMHMM_MODEL", str(Path(__file__).parent / "tmhmm" / "TMHMM2.0.model"))).resolve()
ENABLE_GOR4 = env_bool("ENABLE_GOR4", True)
GOR4_CACHE_DIR = os.getenv("GOR4_CACHE_DIR", f"{CACHE_DIR}/gor4")
ANNOTATION_TARGET = os.getenv("ANNOTATION_TARGET", "outliers")
PIPELINE_STAGE_COUNT = 8

for d in [OUTPUT_DIR, CACHE_DIR, GENE_DIR, UP_GENES_DIR, DOWN_GENES_DIR,
        NETWORK_DIR, UP_NETWORK, DOWN_NETWORK, TMHMM_CACHE_DIR, GOR4_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


def print_stage(index, name):
    print(f"\n[STAGE {index}/{PIPELINE_STAGE_COUNT}] {name}")

print("=" * 60)
print("TRANSCRIPTOMICS ANALYSIS PIPELINE - UNIFIED VERSION")
print("=" * 60)
print(
    f"Active species: {PLANT_SPECIES_LABEL} ({SPECIES_SCIENTIFIC_NAME}) | "
    f"UniProt tax ID: {UNIPROT_TAX_ID} | STRING species: {STRING_SPECIES}"
)

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


def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""


def read_expression_file(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".xls":
        return pd.read_excel(path, engine="xlrd")
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_excel(path)

def create_session():
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=0, pool_connections=20, pool_maxsize=20)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    default_headers = {
        "User-Agent": f"tap-pipeline/1.0 (+client-ip:{REQUEST_CLIENT_IP})",
        "X-Client-IP": REQUEST_CLIENT_IP,
    }
    if REQUEST_CLIENT_API_KEY:
        default_headers["X-Client-Api-Key"] = REQUEST_CLIENT_API_KEY
    session.headers.update(default_headers)
    return session

SESSION = create_session()

def safe_request(url, params=None):
    try:
        r = SESSION.get(url, params=params)
        if r.ok and r.text.strip():
            return r
    except Exception:
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
        r = safe_request(url)
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
    if not seq:
        return ""
    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    return "".join([aa for aa in seq if aa in VALID_AA])


def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def run_tmhmm_local(seq: str, gene_dir, gene_id: str, force: bool = False):
    gene_dir_path = Path(gene_dir)
    gene_dir_path.mkdir(parents=True, exist_ok=True)
    txt_path = gene_dir_path / f"tmhmm_{gene_id}.txt"
    png_path = gene_dir_path / f"tmhmm_{gene_id}.png"
    summary_path = gene_dir_path / f"tmhmm_{gene_id}.summary"
    annotation_path = gene_dir_path / f"tmhmm_{gene_id}.annotation"
    posterior_path = gene_dir_path / f"tmhmm_{gene_id}.plot"

    sequence = clean_protein_sequence(seq)
    if len(sequence) < 20:
        return None

    try:
        from tmhmm.cli import has_matplotlib

        if not TMHMM_MODEL.exists():
            print(f"  [TMHMM] model not found at {TMHMM_MODEL}")
            return None
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", dir=gene_dir_path, delete=False, encoding="utf-8"
        ) as fasta_file:
            fasta_file.write(f">{gene_id} TMHMM sequence\n{sequence}\n")
            fasta_path = Path(fasta_file.name)

        cli_args = ["-f", str(fasta_path), "-m", str(TMHMM_MODEL)]
        if has_matplotlib:
            cli_args.append("-p")
        cli_env = os.environ.copy()
        existing_pythonpath = cli_env.get("PYTHONPATH", "")
        cli_env["PYTHONPATH"] = os.pathsep.join(
            value for value in [str(TMHMM_PACKAGE_ROOT), existing_pythonpath] if value
        )
        result = subprocess.run(
            [sys.executable, "-c", "from tmhmm.cli import cli; cli()", *cli_args],
            capture_output=True,
            text=True,
            cwd=str(gene_dir_path),
            env=cli_env,
            timeout=120,
        )
        if result.returncode:
            print(f"  [TMHMM] {gene_id}: CLI failed: {(result.stderr or result.stdout)[-500:]}")
            return None

        generated_files = {
            gene_dir_path / f"{gene_id}.summary": summary_path,
            gene_dir_path / f"{gene_id}.annotation": annotation_path,
            gene_dir_path / f"{gene_id}.plot": posterior_path,
            gene_dir_path / f"{gene_id}.png": png_path,
        }
        for generated, destination in generated_files.items():
            if generated.exists():
                generated.replace(destination)

        summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        summary_lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
        membrane_segments = []
        for line in summary_lines:
            parts = line.split(maxsplit=2)
            if len(parts) == 3 and parts[2].lower() == "transmembrane helix":
                start, end = int(parts[0]), int(parts[1])
                membrane_segments.append({"start": start + 1, "end": end + 1})
        segment_text = ", ".join(
            f"{segment['start']}-{segment['end']}" for segment in membrane_segments
        ) or "None"
        annotation_text = annotation_path.read_text(encoding="utf-8") if annotation_path.exists() else ""
        topology_path = "".join(
            line.strip() for line in annotation_text.splitlines() if line and not line.startswith(">")
        )
        topology = "; ".join(summary_lines)
        output = (
            f"TMHMM 2.0 prediction for {gene_id}\n"
            f"Sequence length: {len(sequence)} aa\n"
            f"Transmembrane helices: {len(membrane_segments)}\n"
            f"Helix positions (1-based): {segment_text}\n"
            f"Topology path: {topology_path}\n"
        )
        txt_path.write_text(output, encoding="utf-8")

        return {
            "text": output,
            "text_path": txt_path,
            "summary_path": summary_path,
            "annotation_path": annotation_path,
            "posterior_path": posterior_path,
            "png_path": png_path if png_path.exists() else None,
            "segments": membrane_segments,
            "topology": topology,
            "cached": False,
        }
    except Exception as exc:
        print(f"  [TMHMM] {gene_id}: prediction failed: {exc}")
        return None
    finally:
        if "fasta_path" in locals():
            fasta_path.unlink(missing_ok=True)


_GOR4_PREDICTOR = None


def run_secondary_structure_local(seq, gene_dir=None, gene_id=None):
    global _GOR4_PREDICTOR
    if not ENABLE_GOR4:
        return None
    seq = clean_protein_sequence(seq)
    if len(seq) < 10:
        return None

    gene_id = gene_id or "protein"
    reports_dir = Path(gene_dir) if gene_dir else Path(CACHE_DIR) / "gor4"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(GOR4_CACHE_DIR) / f"{_seq_hash(seq)}.json"
    raw_path = reports_dir / f"{gene_id}_gor4_raw.txt"
    probability_path = reports_dir / f"secondary_{gene_id}.csv"

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        if not cached or cached.get("sequence") != seq:
            from gor4 import GOR4

            if _GOR4_PREDICTOR is None:
                _GOR4_PREDICTOR = GOR4()
            prediction = _GOR4_PREDICTOR.predict(seq)
            cached = {
                "sequence": seq,
                "states": prediction["predictions"],
                "probabilities": [list(map(float, values)) for values in prediction["probabilities"]],
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached), encoding="utf-8")

        states = cached["states"]
        probabilities = cached["probabilities"]
        if len(states) != len(seq) or len(probabilities) != len(seq):
            raise ValueError("GOR4 returned a result with an unexpected sequence length")

        formatted_lines = ["Position|AA|Prediction|P(H)|P(E)|P(C)"]
        with probability_path.open("w", newline="", encoding="utf-8") as probability_file:
            writer = csv.writer(probability_file)
            writer.writerow(["position", "amino_acid", "prediction", "p_helix", "p_strand", "p_coil"])
            for position, (amino_acid, state, values) in enumerate(zip(seq, states, probabilities), 1):
                helix, strand, coil = values
                writer.writerow([position, amino_acid, state, helix, strand, coil])
                formatted_lines.append(
                    f"{position}|{amino_acid}|{state}|{helix:.4f}|{strand:.4f}|{coil:.4f}"
                )
        raw_path.write_text("\n".join(formatted_lines) + "\n", encoding="utf-8")

        confidence = [
            values[{"H": 0, "E": 1, "C": 2}[state]]
            for state, values in zip(states, probabilities)
        ]
        return {
            "states": states,
            "confidence": confidence,
            "probabilities": probabilities,
            "formatted_lines": formatted_lines,
            "method": "local GOR IV",
            "gor4_artifacts": {"gor4_raw": str(raw_path), "secondary_probabilities": str(probability_path)},
        }
    except Exception as exc:
        print(f"  [GOR4] local prediction failed for {gene_id}: {exc}")
        return None

annotation_cache = load_cache("annotations.json")
uniprot_cache = load_cache("uniprot_search.json")
eggnog_cache = load_cache("eggnog.json")
kegg_cache = load_cache("kegg.json")
alphafold_cache = load_cache("alphafold.json")
uniprot_entry_cache = load_cache("uniprot_entry.json")
string_tsv_cache = load_cache("string_tsv.json")
string_id_cache = load_cache("string_ids.json")
string_cytoscape_cache = {}
gene_panel_cache = {}


start_time = time.time()

def _split_by_direction(target_df):
    up = target_df[target_df["log2FC"] >= LOG2FC_THRESHOLD].copy()
    down = target_df[target_df["log2FC"] <= -LOG2FC_THRESHOLD].copy()
    return up, down

print_stage(1, "Loading and filtering expression data")

if is_step_complete_with_files("data_loading", [
    f"{CACHE_DIR}/processed_data.csv",
    f"{OUTPUT_DIR}/upregulated_genes.csv",
    f"{OUTPUT_DIR}/downregulated_genes.csv",
    f"{OUTPUT_DIR}/outlier_genes.csv"
]):
    print("  [OK] Using cached data")
    try:
        raw_total = len(read_expression_file(EXPR_FILE))
    except Exception:
        raw_total = None
    df = pd.read_csv(f"{CACHE_DIR}/processed_data.csv")
    upregulated = pd.read_csv(f"{OUTPUT_DIR}/upregulated_genes.csv")
    downregulated = pd.read_csv(f"{OUTPUT_DIR}/downregulated_genes.csv")
    outliers = pd.read_csv(f"{OUTPUT_DIR}/outlier_genes.csv")
    filtered_total = len(df)
    pval_column = "adj_p_value" if USE_ADJUSTED_PVALUE else "p_value"
    if pval_column not in df.columns:
        pval_column = "p_value" if "p_value" in df.columns else pval_column
    sig_up = df[(df["log2FC"] >= LOG2FC_THRESHOLD) & (df[pval_column] <= PVALUE_THRESHOLD)]
    sig_down = df[(df["log2FC"] <= -LOG2FC_THRESHOLD) & (df[pval_column] <= PVALUE_THRESHOLD)]
    significant = pd.concat([sig_up, sig_down]).drop_duplicates(subset=["gene_id"])
else:
    print("  Processing expression data...")
    raw_df = read_expression_file(EXPR_FILE)
    raw_total = len(raw_df)
    df = raw_df.copy()
    required_cols = ["gene_id", "log2FC", "p_value"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=required_cols)
    df = df[df["p_value"] > 0]

    min_abs_log2fc = 0.1
    max_pval = 0.99
    df = df[(df["log2FC"].abs() >= min_abs_log2fc) & (df["p_value"] <= max_pval)]
    filtered_total = len(df)
    if df.empty:
        raise ValueError("No valid data after low-expression filtering")

    existing_adj = "adj_p_value" in df.columns
    if existing_adj:
        df["adj_p_value"] = pd.to_numeric(df["adj_p_value"], errors="coerce")
        invalid_adj = df["adj_p_value"].isna() | (df["adj_p_value"] <= 0)
    else:
        invalid_adj = pd.Series(True, index=df.index)

    if (not existing_adj) or invalid_adj.any():
        _, adj_pvals, _, _ = multipletests(df["p_value"], method="fdr_bh")
        adj_series = pd.Series(adj_pvals, index=df.index)
        if existing_adj:
            df.loc[invalid_adj, "adj_p_value"] = adj_series.loc[invalid_adj]
        else:
            df["adj_p_value"] = adj_series

    df["adj_p_value"] = df["adj_p_value"].clip(lower=1e-300)
    df.to_csv(os.path.join(OUTPUT_DIR, "upload.csv"), index=False)

    pval_column = "adj_p_value" if USE_ADJUSTED_PVALUE else "p_value"
    df[pval_column] = df[pval_column].clip(lower=1e-300)
    df["neg_log10_p"] = -np.log10(df[pval_column])
    df["euclidean_distance"] = np.sqrt(df["log2FC"]**2 + df["neg_log10_p"]**2)

    sig_up = df[(df["log2FC"] >= LOG2FC_THRESHOLD) & (df[pval_column] <= PVALUE_THRESHOLD)]
    sig_down = df[(df["log2FC"] <= -LOG2FC_THRESHOLD) & (df[pval_column] <= PVALUE_THRESHOLD)]
    significant = pd.concat([sig_up, sig_down]).drop_duplicates(subset=["gene_id"])

    if not significant.empty:
        cutoff = np.percentile(significant["euclidean_distance"], DISTANCE_PERCENTILE)
        outliers = significant[significant["euclidean_distance"] >= cutoff].copy()
    else:
        outliers = significant.copy()

    target_df = significant if ANNOTATION_TARGET == "all" else outliers
    upregulated, downregulated = _split_by_direction(target_df)

    df.to_csv(f"{CACHE_DIR}/processed_data.csv", index=False)
    upregulated.to_csv(f"{OUTPUT_DIR}/upregulated_genes.csv", index=False)
    downregulated.to_csv(f"{OUTPUT_DIR}/downregulated_genes.csv", index=False)
    outliers.to_csv(f"{OUTPUT_DIR}/outlier_genes.csv", index=False)
    mark_step_complete("data_loading")

print(f"  Total genes in data : {raw_total if raw_total is not None else 'N/A'}")
print(f"  no. of significant upregulated genes: {len(sig_up)}")
print(f"  no. of significant downregulated genes: {len(sig_down)}")
print(f"  no. of genes after filtering log2fc and p value : {filtered_total}")
print(f"  annotation target: {ANNOTATION_TARGET}")
print(f"  no. of genes selected for annotation (upregulated): {len(upregulated)}")
print(f"  no. of genes selected for annotation (downregulated): {len(downregulated)}")

def uniprot_search(gene_id, cache):
    if gene_id in cache:
        return cache[gene_id]
    
    tax_filter = f"(organism_id:{UNIPROT_TAX_ID})"
    queries = [
        f"(gene_exact:{gene_id}) AND {tax_filter}",
        f"(gene:{gene_id}) AND {tax_filter}",
        f"{gene_id} AND {tax_filter}"
    ]
                                                                                             
    queries.extend([
        f"gene_exact:{gene_id}",
        f"gene:{gene_id}",
        f"{gene_id}",
    ])
    
    for query in queries:
        params = {"query": query, "format": "json", "size": 1}
        
        r = safe_request(UNIPROT_SEARCH, params=params)
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

    r = safe_request(f"{UNIPROT_ENTRY}/{acc}.json")
    data = safe_json(r)
    uniprot_entry_cache[acc] = data
    return data

def get_string_tsv(gene_id, species=None, fallback_ids=None):
    species = species or STRING_SPECIES
    cache_key = f"{gene_id}_{species}"
    if cache_key in string_tsv_cache:
        return string_tsv_cache[cache_key]

    resolved_id = resolve_string_identifier(gene_id, species=species, fallback_ids=fallback_ids)
    url = f"{STRING_API}/tsv/network?identifiers={resolved_id}&species={species}"
    params = {"required_score": 400}
    r_net = safe_request(url, params=params)

    if not r_net or not r_net.text.strip():
        string_tsv_cache[cache_key] = None
        return None

    string_tsv_cache[cache_key] = r_net.text
    return r_net.text


def get_string_ids(identifiers, species=None):
    species = species or STRING_SPECIES
    if not identifiers:
        return {}

    if isinstance(identifiers, str):
        identifiers = [identifiers]

    result = {}
    to_query = []
    for ident in identifiers:
        cache_key = f"{species}:{ident}"
        if cache_key in string_id_cache:
            result[ident] = string_id_cache.get(cache_key)
        else:
            to_query.append(ident)

    if to_query:
        ids_string = "%0d".join(to_query)
        url = f"{STRING_API}/tsv-no-header/get_string_ids"
        params = {"identifiers": ids_string, "species": species, "limit": 1}
        r = safe_request(url, params=params)
        if r and r.text.strip():
            for line in r.text.strip().split("\n"):
                cols = line.split("\t")
                if len(cols) >= 2:
                    query = cols[0]
                    string_id = cols[1]
                    result[query] = string_id
                    string_id_cache[f"{species}:{query}"] = string_id

        for ident in to_query:
            if ident not in result:
                result[ident] = None
                string_id_cache[f"{species}:{ident}"] = None

    return result


def resolve_string_identifier(gene_id, species=None, fallback_ids=None):
    candidates = [gene_id]
    if fallback_ids:
        candidates.extend([c for c in fallback_ids if c])
    mapping = get_string_ids(candidates, species=species)
    for ident in candidates:
        string_id = mapping.get(ident)
        if string_id:
            return string_id
    return gene_id


def get_string_network_cytoscape(gene_id, species=None):
    species = species or STRING_SPECIES
    cache_key = f"{gene_id}_{species}"
    if cache_key in string_cytoscape_cache:
        return string_cytoscape_cache[cache_key]

    resolved_id = resolve_string_identifier(gene_id, species=species)
    api_url = f"{STRING_API}/tsv/network"
    params = {"identifiers": resolved_id, "species": species, "required_score": 400}
    
    r = safe_request(api_url, params=params)
    
    if not r or not r.text.strip():
        return [], []
    
    lines = r.text.strip().split("\n")[1:]
    if not lines:
        return [], []
    
                          
    G = nx.Graph()
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 6:
            protein1, protein2, score = parts[2], parts[3], float(parts[5])
            G.add_edge(protein1, protein2, weight=score)
    
    if len(G.edges) == 0:
        return [], []
    
                     
    partition = community.best_partition(G)
    
    cluster_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
    ]
    
                     
    query_node_id = None
    gene_id_upper = gene_id.upper()

    for node in G.nodes():
        node_upper = node.upper()

        if gene_id_upper in node_upper:
            query_node_id = node
            break


    
                   
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
    
                   
    edges = [{'data': {'source': u, 'target': v, 'weight': d['weight']}} 
             for u, v, d in G.edges(data=True)]
    
    string_cytoscape_cache[cache_key] = (nodes, edges)
    return nodes, edges

                                                            
                      
                                                            
print_stage(2, "UniProt search")

annotations_path = Path(OUTPUT_DIR) / "annotations.csv"
cached_annotation_columns = set()
if annotations_path.exists():
    try:
        cached_annotation_columns = set(pd.read_csv(annotations_path, nrows=0).columns)
    except Exception:
        cached_annotation_columns = set()

if "uniprot_details" in cached_annotation_columns and is_step_complete_with_files("unified_annotation", [
    f"{OUTPUT_DIR}/annotations.csv",
    f"{OUTPUT_DIR}/kegg_pathways.csv",
    f"{OUTPUT_DIR}/eggnog_go_terms.csv",
    f"{OUTPUT_DIR}/eggnog_domains.csv",
    f"{OUTPUT_DIR}/interpro_domains.csv"
]):
    print("  [OK] Using cached annotations")
    ann_df = pd.read_csv(f"{OUTPUT_DIR}/annotations.csv")
    kegg_df = pd.read_csv(f"{OUTPUT_DIR}/kegg_pathways.csv")
    eggnog_go_df = pd.read_csv(f"{OUTPUT_DIR}/eggnog_go_terms.csv")
    eggnog_domains_df = pd.read_csv(f"{OUTPUT_DIR}/eggnog_domains.csv")
    interpro_df = pd.read_csv(f"{OUTPUT_DIR}/interpro_domains.csv")
else:
    annotation_target_val = os.getenv("ANNOTATION_TARGET", "outliers")
    if annotation_target_val == "all" and "sig_up" in locals() and "sig_down" in locals():
        all_genes = pd.concat([sig_up, sig_down]).drop_duplicates(subset=["gene_id"])
    else:
        if "outliers" in locals() and isinstance(outliers, pd.DataFrame) and not outliers.empty:
            all_genes = outliers
        else:
            all_genes = pd.concat([upregulated, downregulated]).drop_duplicates(subset=["gene_id"])

    gene_ids_list = all_genes["gene_id"].tolist()
    total_genes = len(gene_ids_list)
    print(f"  Annotating {total_genes} genes in bulk...")

    print("  UniProt search...")

    def _bulk_uniprot_search_batch(batch_genes):
        gene_query = " OR ".join(f"gene:{g}" for g in batch_genes)
        query = f"({gene_query}) AND (organism_id:{UNIPROT_TAX_ID})"
        params = {"query": query, "format": "json", "size": 500}
        try:
            r = SESSION.get(UNIPROT_SEARCH, params=params, timeout=60)
            if r.ok:
                return r.json().get("results", [])
        except Exception:
            pass
        return []

    def _bulk_uniprot_no_taxon_batch(batch_genes):
        gene_query = " OR ".join(f"gene:{g}" for g in batch_genes)
        params = {"query": gene_query, "format": "json", "size": 500}
        try:
            r = SESSION.get(UNIPROT_SEARCH, params=params, timeout=60)
            if r.ok:
                return r.json().get("results", [])
        except Exception:
            pass
        return []

    BATCH_SIZE = 50
    all_uniprot_entries = []
    batches = [gene_ids_list[i:i+BATCH_SIZE] for i in range(0, len(gene_ids_list), BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_bulk_uniprot_search_batch, b): b for b in batches}
        for f in as_completed(futures):
            try:
                all_uniprot_entries.extend(f.result())
            except Exception:
                pass

    found_genes = set()
    for entry in all_uniprot_entries:
        for gd in entry.get("genes", []):
            gname = gd.get("geneName", {}).get("value", "")
            if gname:
                found_genes.add(gname.upper())

    missing_genes = [g for g in gene_ids_list if g.upper() not in found_genes]
    if missing_genes:
        miss_batches = [missing_genes[i:i+BATCH_SIZE] for i in range(0, len(missing_genes), BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_bulk_uniprot_no_taxon_batch, b): b for b in miss_batches}
            for f in as_completed(futures):
                try:
                    all_uniprot_entries.extend(f.result())
                except Exception:
                    pass

    print(f"  Found {len(all_uniprot_entries)} UniProt entries")

    def _extract_uniprot_annotation(entry):
        acc = entry.get("primaryAccession", "")
        protein = entry.get("proteinDescription", {})
        rec_name = protein.get("recommendedName", {})
        protein_name = rec_name.get("fullName", {}).get("value", "Unknown")
        gene_name = ""
        gene_names = []
        genes_data = entry.get("genes", [])
        for g_data in genes_data:
            if "geneName" in g_data:
                gene_names.append(g_data["geneName"].get("value", ""))
            for syn in g_data.get("synonyms", []):
                gene_names.append(syn.get("value", ""))
            for orf in g_data.get("orfNames", []):
                gene_names.append(orf.get("value", ""))
            for ol in g_data.get("orderedLocusNames", []):
                gene_names.append(ol.get("value", ""))
        gene_names = [n for n in gene_names if n]
        gene_name = gene_names[0] if gene_names else ""
        
        organism_name = entry.get("organism", {}).get("scientificName", "")
        protein_length = entry.get("sequence", {}).get("length", 0)
        molecular_weight = entry.get("sequence", {}).get("mass", 0)
        sequence = entry.get("sequence", {}).get("value", "")
        function_texts = []
        locations = []
        for comment in entry.get("comments", []):
            ctype = comment.get("commentType", "")
            if ctype == "FUNCTION":
                for t in comment.get("texts", []):
                    function_texts.append(t.get("value", ""))
            elif ctype == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    loc_val = loc.get("location", {}).get("value", "")
                    if loc_val:
                        locations.append(loc_val)
        go_bp, go_mf, go_cc = [], [], []
        interpro_ids, pfam_ids, cath_ids, eggnog_ids = [], [], [], []
        kegg_ids, panther_ids = [], []
        for xref in entry.get("uniProtKBCrossReferences", []):
            db = xref.get("database", "")
            db_id = xref.get("id", "")
            if db == "GO":
                props = xref.get("properties", [])
                term = next((p.get("value", "") for p in props if p.get("key") == "GoTerm"), "")
                if term.startswith("P:"):
                    go_bp.append(f"{db_id} ({term[2:]})")
                elif term.startswith("F:"):
                    go_mf.append(f"{db_id} ({term[2:]})")
                elif term.startswith("C:"):
                    go_cc.append(f"{db_id} ({term[2:]})")
            elif db == "InterPro":
                interpro_ids.append(db_id)
            elif db == "Pfam":
                pfam_ids.append(db_id)
            elif db == "Gene3D":
                cath_ids.append(db_id)
            elif db == "eggNOG":
                eggnog_ids.append(db_id)
            elif db == "KEGG":
                kegg_ids.append(db_id)
            elif db == "PANTHER":
                panther_ids.append(db_id)
                
        keywords = "; ".join(x.get("name", "") for x in entry.get("keywords", []))
        references = {"PubMed", "DOI"}
        useful_cross_references = [
            xref for xref in entry.get("uniProtKBCrossReferences", [])
            if xref.get("database") not in references
        ]
        excluded_comment_types = {"MASS SPECTROMETRY", "WEB RESOURCE"}
        useful_comments = [
            comment for comment in entry.get("comments", [])
            if comment.get("commentType") not in excluded_comment_types
        ]
        sequence_data = entry.get("sequence", {})
        uniprot_details = {
            "entryType": entry.get("entryType"),
            "primaryAccession": acc,
            "uniProtkbId": entry.get("uniProtkbId"),
            "entryAudit": entry.get("entryAudit", {}),
            "annotationScore": entry.get("annotationScore"),
            "organism": entry.get("organism", {}),
            "proteinExistence": entry.get("proteinExistence"),
            "proteinDescription": protein,
            "genes": genes_data,
            "comments": useful_comments,
            "features": entry.get("features", []),
            "keywords": entry.get("keywords", []),
            "crossReferences": useful_cross_references,
            "sequence": sequence_data,
            "extraAttributes": entry.get("extraAttributes", {}),
        }
        return {
            "acc": acc,
            "gene_name": gene_name,
            "all_gene_names": gene_names,
            "organism": organism_name,
            "protein_name": protein_name,
            "function": " | ".join(function_texts)[:800],
            "subcellular_location": "; ".join(locations),
            "protein_length": protein_length,
            "molecular_weight": molecular_weight,
            "sequence": sequence,
            "GO_biological_process": "; ".join(go_bp[:10]),
            "GO_molecular_function": "; ".join(go_mf[:10]),
            "GO_cellular_component": "; ".join(go_cc[:10]),
            "InterPro_domains": "; ".join(interpro_ids[:10]),
            "Pfam_domains": "; ".join(pfam_ids[:10]),
            "CATH_domains": "; ".join(cath_ids[:10]),
            "KEGG": "; ".join(kegg_ids[:10]),
            "PANTHER": "; ".join(panther_ids[:10]),
            "eggnog_ids": eggnog_ids,
            "keywords": keywords,
            "uniprot_details": uniprot_details,
        }

    uniprot_map = {}
    gene_to_acc = {}
    for entry in all_uniprot_entries:
        ann = _extract_uniprot_annotation(entry)
        acc = ann["acc"]
        if acc and acc not in uniprot_map:
            uniprot_map[acc] = ann
        for gn in ann.get("all_gene_names", []):
            gn_up = gn.upper()
            if gn_up and gn_up not in gene_to_acc:
                gene_to_acc[gn_up] = acc

    for gene_id in gene_ids_list:
        if gene_id.upper() not in gene_to_acc:
            for acc, ann in uniprot_map.items():
                if any(gene_id.upper() in gn.upper() for gn in ann.get("all_gene_names", [])):
                    gene_to_acc[gene_id.upper()] = acc
                    break

    save_cache("uniprot_search.json", {g: gene_to_acc.get(g.upper()) for g in gene_ids_list})

    def _download_alphafold_pdb_bulk(uniprot_id, out_path):
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return uniprot_id, "EXISTS"
        url = f"{ALPHAFOLD_BASE}/AF-{uniprot_id}-F1-model_v6.pdb"
        try:
            r = SESSION.get(url, timeout=15)
            if r.status_code == 200 and r.content:
                with open(out_path, "wb") as fh:
                    fh.write(r.content)
                return uniprot_id, "DOWNLOADED"
            return uniprot_id, f"HTTP_{r.status_code}"
        except Exception:
            return uniprot_id, "ERROR"

    def _get_alphafold_meta_bulk(uniprot_id):
        cache_key = f"af_{uniprot_id}"
        if cache_key in alphafold_cache:
            return alphafold_cache[cache_key]
        try:
            r = SESSION.get(f"{ALPHAFOLD_API}/{uniprot_id}", timeout=10)
            data = r.json() if r.ok else None
            if data and len(data) > 0:
                af = data[0]
                result = {
                    "model_version": af.get("latestVersion", af.get("modelVersion", "Unknown")),
                    "confidence": af.get("globalMetricValue", "N/A"),
                    "pdb_url": af.get("pdbUrl", ""),
                    "created_date": af.get("modelCreatedDate", "Unknown"),
                }
                alphafold_cache[cache_key] = result
                return result
        except Exception:
            pass
        alphafold_cache[cache_key] = None
        return None

    gene_dir_map = {}
    for gene_id in gene_ids_list:
        row = all_genes[all_genes["gene_id"] == gene_id]
        log2fc = row.iloc[0]["log2FC"] if not row.empty else 0
        base_dir = UP_GENES_DIR if log2fc > 0 else DOWN_GENES_DIR
        gene_dir = f"{base_dir}/{gene_id}"
        os.makedirs(gene_dir, exist_ok=True)
        gene_dir_map[gene_id] = gene_dir

    fasta_seq_map = {}
    for gene_id in gene_ids_list:
        acc = gene_to_acc.get(gene_id.upper())
        uni_ann = uniprot_map.get(acc, {}) if acc else {}
        seq = clean_protein_sequence(uni_ann.get("sequence", ""))
        if not seq:
            continue
        fasta_seq_map[gene_id] = seq
        gene_dir_local = gene_dir_map.get(gene_id, "")
        if gene_dir_local:
            fasta_path = f"{gene_dir_local}/protein_{gene_id}.fasta"
            if not (os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0):
                try:
                    with open(fasta_path, "w", encoding="utf-8") as fh:
                        fh.write(f">{acc or gene_id}|{gene_id}\n{seq}\n")
                except Exception:
                    pass

    print_stage(3, "AlphaFold structures")

    acc_to_gene = {}
    for gene_id in gene_ids_list:
        acc = gene_to_acc.get(gene_id.upper())
        if acc and acc not in acc_to_gene:
            acc_to_gene[acc] = gene_id

    def _af_worker(acc):
        gene_id = acc_to_gene.get(acc, acc)
        gene_dir = gene_dir_map.get(gene_id, f"{UP_GENES_DIR}/{gene_id}")
        os.makedirs(gene_dir, exist_ok=True)
        pdb_path = f"{gene_dir}/structure_{gene_id}.pdb"
        uid, status = _download_alphafold_pdb_bulk(acc, pdb_path)
        meta = _get_alphafold_meta_bulk(acc)
        return acc, status, meta, pdb_path

    unique_accs = list(set(v for v in gene_to_acc.values() if v))
    af_results = {}
    af_meta = {}

    with ThreadPoolExecutor(max_workers=20) as pool:
        af_futures = {pool.submit(_af_worker, acc): acc for acc in unique_accs}
        done = 0
        for fut in as_completed(af_futures):
            done += 1
            try:
                acc, status, meta, pdb_path = fut.result()
                af_results[acc] = {"status": status, "pdb_path": pdb_path}
                af_meta[acc] = meta
                print(f"  AF [{done}/{len(unique_accs)}] {acc}: {status}")
            except Exception:
                pass

    save_cache("alphafold.json", alphafold_cache)

    print_stage(4, "Secondary structure prediction (local GOR IV)")

    def _secondary_worker(args):
        gene_id, seq, gene_dir, acc = args
        if not seq or len(seq) < 10:
            return gene_id, None
        try:
            result = run_secondary_structure_local(seq, gene_dir=gene_dir, gene_id=gene_id)
            return gene_id, result
        except Exception:
            return gene_id, None

    secondary_tasks = [
        (gene_id, fasta_seq_map.get(gene_id, ""), gene_dir_map.get(gene_id, ""), gene_to_acc.get(gene_id.upper()))
        for gene_id in gene_ids_list
        if fasta_seq_map.get(gene_id)
    ]

    secondary_map = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        sec_futures = {pool.submit(_secondary_worker, t): t for t in secondary_tasks}
        for fut in as_completed(sec_futures):
            try:
                gene_id, result = fut.result()
                secondary_map[gene_id] = result
            except Exception:
                pass

    print_stage(5, "TMHMM topology prediction (local)")

    def _tmhmm_worker(args):
        gene_id, seq, gene_dir = args
        if not seq or len(seq) < 20:
            return gene_id, None
        try:
            result = run_tmhmm_local(seq, gene_dir, gene_id)
            return gene_id, result
        except Exception:
            return gene_id, None

    tmhmm_tasks = [
        (gene_id, fasta_seq_map.get(gene_id, ""), gene_dir_map.get(gene_id, ""))
        for gene_id in gene_ids_list
        if fasta_seq_map.get(gene_id)
    ]

    tmhmm_map = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        tmhmm_futures = {pool.submit(_tmhmm_worker, t): t for t in tmhmm_tasks}
        for fut in as_completed(tmhmm_futures):
            try:
                gene_id, result = fut.result()
                tmhmm_map[gene_id] = result
            except Exception:
                pass

    print_stage(6, "STRING networks")

    kegg_records = []

    def _string_worker(args):
        gene_id, acc, gene_dir = args
        string_identifier = resolve_string_identifier(gene_id, species=STRING_SPECIES, fallback_ids=[acc] if acc else [])
        try:
            r_svg = SESSION.get(
                f"{STRING_API}/svg/network?identifiers={string_identifier}&species={STRING_SPECIES}",
                timeout=15
            )
            if r_svg.ok and r_svg.content:
                with open(f"{gene_dir}/string_{gene_id}.svg", "wb") as fh:
                    fh.write(r_svg.content)
        except Exception:
            pass
        edges_out = []
        try:
            r_net = SESSION.get(
                f"{STRING_API}/tsv/network?identifiers={string_identifier}&species={STRING_SPECIES}",
                params={"required_score": 400},
                timeout=15
            )
            if r_net.ok and r_net.text.strip():
                for line in r_net.text.strip().split("\n")[1:]:
                    cols = line.split("\t")
                    if len(cols) >= 6:
                        try:
                            edges_out.append({
                                "source": cols[2],
                                "target": cols[3],
                                "combined_score": float(cols[5])
                            })
                        except Exception:
                            pass
                if edges_out:
                    pd.DataFrame(edges_out).to_csv(f"{gene_dir}/string_edges_{gene_id}.csv", index=False)
        except Exception:
            pass
        return gene_id, edges_out

    string_tasks = [
        (gene_id, gene_to_acc.get(gene_id.upper()), gene_dir_map.get(gene_id, ""))
        for gene_id in gene_ids_list if gene_dir_map.get(gene_id)
    ]

    string_edge_map = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        str_futures = {pool.submit(_string_worker, t): t for t in string_tasks}
        for fut in as_completed(str_futures):
            try:
                gene_id, edges = fut.result()
                string_edge_map[gene_id] = edges
            except Exception:
                pass

    for gene_id in gene_ids_list:
        acc = gene_to_acc.get(gene_id.upper())
        uni_ann = uniprot_map.get(acc, {}) if acc else {}
        kegg_ids = [x for x in uni_ann.get("KEGG", "").split("; ") if x]
        for kegg_id in kegg_ids:
            kegg_records.append({
                "gene_id": gene_id,
                "uniprot_id": acc or "",
                "KEGG_gene_id": kegg_id,
                "KO": "",
                "KEGG_pathway_id": kegg_id,
                "KEGG_pathway_name": "",
                "KEGG_pathway_class": "",
                "KEGG_description": "",
            })

    annotation_records = []

    for _, row in all_genes.iterrows():
        gene_id = row["gene_id"]
        log2fc = row["log2FC"]
        pvalue = row["p_value"]
        acc = gene_to_acc.get(gene_id.upper())
        uni_ann = uniprot_map.get(acc, {}) if acc else {}
        af_m = af_meta.get(acc) if acc else None
        tmhmm_result = tmhmm_map.get(gene_id)

        tmhmm_text = ""
        tmhmm_plot = ""
        tmhmm_is_transporter = False
        tmhmm_segments = []
        tmhmm_topology = ""
        if tmhmm_result:
            tmhmm_text = tmhmm_result.get("text", "")
            tmhmm_segments = tmhmm_result.get("segments", [])
            tmhmm_topology = tmhmm_result.get("topology", "")
            png = tmhmm_result.get("png_path")
            tmhmm_plot = str(png) if png else ""
            tmhmm_is_transporter = bool(tmhmm_segments)

        seq = fasta_seq_map.get(gene_id, "")
        sec_pred = secondary_map.get(gene_id) or {"states": "", "confidence": [], "formatted_lines": [], "method": "none"}

        gene_dir_local = gene_dir_map.get(gene_id, "")
        if gene_dir_local and (sec_pred.get("states") or sec_pred.get("confidence")):
            try:
                with open(f"{gene_dir_local}/secondary_{gene_id}.txt", "w") as fh:
                    formatted = sec_pred.get("formatted_lines") or []
                    if formatted:
                        fh.write("\n".join(formatted))
                    else:
                        fh.write(f"Sequence\n{seq}\n")
                        fh.write("|" * len(seq) + "\n")
                        fh.write(sec_pred.get("states", ""))
            except Exception:
                pass

        eggnog_id_list = uni_ann.get("eggnog_ids", [])
        orth_groups = "; ".join([f"{e} (EggNOG/COG)" for e in eggnog_id_list]) if eggnog_id_list else ""

        subcell = uni_ann.get("subcellular_location", "")
        if subcell:
            place_loc = subcell
        elif tmhmm_is_transporter:
            place_loc = "Membrane (via TM spans)"
        else:
            place_loc = "Cytosolic/peripheral (heuristic)"

        af_confidence = "N/A"
        af_version = "Unknown"
        if af_m:
            af_confidence = af_m.get("confidence", "N/A")
            af_version = af_m.get("model_version", "Unknown")

        record = {
            "gene_id": gene_id,
            "log2FC": log2fc,
            "p_value": pvalue,
            "uniprot_id": acc or "",
            "uniprot_details": json.dumps(uni_ann.get("uniprot_details", {}), ensure_ascii=False),
            "gene_name": uni_ann.get("gene_name", ""),
            "organism": uni_ann.get("organism", ""),
            "protein_name": uni_ann.get("protein_name", "Unknown"),
            "function": uni_ann.get("function", "Not available"),
            "GO_biological_process": uni_ann.get("GO_biological_process", ""),
            "GO_molecular_function": uni_ann.get("GO_molecular_function", ""),
            "GO_cellular_component": uni_ann.get("GO_cellular_component", ""),
            "InterPro_domains": uni_ann.get("InterPro_domains", ""),
            "Pfam_domains": uni_ann.get("Pfam_domains", ""),
            "CATH_domains": uni_ann.get("CATH_domains", ""),
            "KEGG": uni_ann.get("KEGG", ""),
            "PANTHER": uni_ann.get("PANTHER", ""),
            "keywords": uni_ann.get("keywords", ""),
            "subcellular_location": subcell,
            "protein_length": uni_ann.get("protein_length", 0),
            "molecular_weight": uni_ann.get("molecular_weight", 0),
            "alphafold_confidence": af_confidence,
            "alphafold_version": af_version,
            "secondary_structure": sec_pred.get("states", ""),
            "secondary_confidence": json.dumps(sec_pred.get("confidence", [])),
            "secondary_probabilities": json.dumps(sec_pred.get("probabilities", [])),
            "secondary_prediction_method": sec_pred.get("method", "unavailable"),
            "rapid_fold_model": "",
            "rapid_fold_path": "",
            "psipred_states": sec_pred.get("states", ""),
            "psipred_confidence": json.dumps(sec_pred.get("confidence", [])),
            "tmhmm_text": tmhmm_text,
            "tmhmm_segments": json.dumps(tmhmm_segments),
            "tmhmm_topology": tmhmm_topology,
            "tmhmm_plot": tmhmm_plot,
            "tmhmm_is_transporter": tmhmm_is_transporter,
            "orthologous_groups": orth_groups,
            "place_location": place_loc,
        }
        annotation_cache[gene_id] = record
        annotation_records.append(record)

    ann_df = pd.DataFrame(annotation_records)
    ann_df.to_csv(f"{OUTPUT_DIR}/annotations.csv", index=False)

    kegg_df = pd.DataFrame(kegg_records) if kegg_records else pd.DataFrame(columns=[
        "gene_id", "uniprot_id", "KEGG_gene_id", "KO", "KEGG_pathway_id",
        "KEGG_pathway_name", "KEGG_pathway_class", "KEGG_description"
    ])
    kegg_df.to_csv(f"{OUTPUT_DIR}/kegg_pathways.csv", index=False)

    eggnog_go_df = pd.DataFrame(columns=["gene_id", "uniprot_id", "eggnog_id", "GO_category", "GO_ID", "GO_name"])
    eggnog_go_df.to_csv(f"{OUTPUT_DIR}/eggnog_go_terms.csv", index=False)

    eggnog_domains_df = pd.DataFrame(columns=["gene_id", "uniprot_id", "eggnog_id", "domain_type", "domain_name"])
    eggnog_domains_df.to_csv(f"{OUTPUT_DIR}/eggnog_domains.csv", index=False)

    interpro_df = pd.DataFrame(columns=["gene_id", "uniprot_id", "interpro_id", "domain_name"])
    interpro_df.to_csv(f"{OUTPUT_DIR}/interpro_domains.csv", index=False)

    save_cache("annotations.json", annotation_cache)
    save_cache("uniprot_search.json", uniprot_cache)
    save_cache("eggnog.json", eggnog_cache)
    save_cache("kegg.json", kegg_cache)
    save_cache("alphafold.json", alphafold_cache)
    save_cache("string_ids.json", string_id_cache)
    mark_step_complete("unified_annotation")

    print(f"\n  [OK] Annotated {len(annotation_records)} genes")
    print(f"  [OK] KEGG identifiers (from UniProt): {len(kegg_records)}")


                       
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

                               
                    
                               

                   
top_kegg = (
    kegg_df["KEGG_pathway_id"]
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={"index": "KEGG_pathway_id", "KEGG_pathway_id": "Gene_count"})
) if not kegg_df.empty else pd.DataFrame()

                             
top_go_bp = (
    eggnog_go_df[eggnog_go_df["GO_category"] == "BP"]["GO_name"]
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={"index": "GO_Biological_Process", "GO_name": "Gene_count"})
) if not eggnog_go_df.empty else pd.DataFrame()

                                        
string_edges = []

                                                           
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


                                                            
                    
                                                            
print_stage(7, "Building gene co-expression networks")

if is_step_complete_with_files("string_networks", [
    f"{UP_NETWORK}/upregulated_edges.csv",
    f"{DOWN_NETWORK}/downregulated_edges.csv"
]):
    print("  [OK] Using cached networks")
else:
    print("  Building protein-protein interaction networks...")
    
    def build_subnetwork(gene_df, network_folder, filename_prefix):
        gene_list = list(gene_df["gene_id"])
        mapping = get_string_ids(gene_list, species=STRING_SPECIES)
        resolved = [mapping.get(g) or g for g in gene_list]
        genes_string = "%0d".join(resolved)
        
        print(f"    {filename_prefix}: ", end="", flush=True)
        
        
        network_url = f"{STRING_API}/tsv/network?identifiers={genes_string}&species={STRING_SPECIES}&required_score=400"
        r = safe_request(network_url)
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
                                   
                pd.DataFrame(columns=["source", "target", "combined_score"]).to_csv(
                    f"{network_folder}/{filename_prefix}_edges.csv", index=False)
                print("no edges", end=" | ")
        else:
                               
            pd.DataFrame(columns=["source", "target", "combined_score"]).to_csv(
                f"{network_folder}/{filename_prefix}_edges.csv", index=False)
            print("no data", end=" | ")
        
        
        svg_url = f"{STRING_API}/svg/network?identifiers={genes_string}&species={STRING_SPECIES}"
        svg_resp = safe_request(svg_url)
        if svg_resp and svg_resp.content:
            with open(f"{network_folder}/{filename_prefix}_network.svg", "wb") as f:
                f.write(svg_resp.content)
            print("svg saved")
        else:
            print("no svg")

    build_subnetwork(upregulated, UP_NETWORK, "upregulated")
    build_subnetwork(downregulated, DOWN_NETWORK, "downregulated")
    mark_step_complete("string_networks")

                                                            
              
                                                            
print_stage(8, "Building volcano plot")
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
    title={
        "text": f"Interactive Volcano Plot - {PLANT_SPECIES_LABEL} Gene Expression",
        "x": 0.5,
        "xanchor": "center",
    },
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

                                                            
                 
                                                            
if RUN_DASH_SERVER:
    print("\nStarting web server...")
else:
                                               
    pass

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = f"{PLANT_SPECIES_SHORT_NAME} Transcriptomics Pipeline"
app.config.suppress_callback_exceptions = True

app.layout = dbc.Container([
    html.H1(
        f"{PLANT_SPECIES_LABEL} Transcriptomics Analysis Pipeline",
        className="text-center my-4",
        style={"color": "#2C3E50"},
    ),
    
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
                                
            html.Hr(),
            html.H5("STRING Upregulated Network Visualization", className="mb-2"),
            html.Div([
                html.Img(
                    src=f"data:image/svg+xml;base64,{get_base64_image(up_network_img)}",
                    style={"width": "100%", "maxWidth": "800px", "display": "block", "margin": "0 auto"}
                ) if os.path.exists(up_network_img) else html.P("Network visualization not available", className="text-center text-muted",
                                             style={"padding": "50px"})
            ]),
                                         
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

                           
    cyto_nodes, cyto_edges = get_string_network_cytoscape(gene_id)
    
    reg = "Upregulated" if row["log2FC"] > 0 else "Downregulated"
    reg_color = "success" if row["log2FC"] > 0 else "danger"
    
                    
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


                
geo_data_file = os.getenv("EXPR_FILE", os.path.join(OUTPUT_DIR, "GEO_data.csv"))
if geo_data_file and os.path.exists(geo_data_file):
    try:
        os.remove(geo_data_file)
    except:
        pass

if os.path.exists(CACHE_DIR):
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".csv") and f not in ["processed_data.csv"]:
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except:
                pass

                 
end_time = time.time()
total_time = end_time - start_time

print(f"Total time taken to run the analysis >>> {int(total_time // 60)} minutes {int(total_time%60)} sec")
            


if __name__ == "__main__" and RUN_DASH_SERVER:
    print("\n" + "=" * 60)
    print(" SERVER READY!")
    print("=" * 60 + "\n")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8050)
elif __name__ == "__main__":
                                                                     
    pass