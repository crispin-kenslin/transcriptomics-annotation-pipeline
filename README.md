# GEO Transcriptomics Web Application

A full-stack transcriptomics analysis workflow that ingests differential expression tables, annotates every gene with rich biological context, and serves polished reports through a FastAPI-powered dashboard. The pipeline bundles automated data cleaning, functional annotation, secondary-structure prediction (GOR4, AlphaSync fallbacks), membrane topology calls (Phobius), STRING interaction networks, and curated download packages—making it easy to review GEO-sized datasets without juggling notebooks.

## Key Features
- **Unified processing pipeline**: `main_final.py` thresholds expression data, normalizes identifiers, and fans out to UniProt, Ensembl, STRING, KEGG, EggNOG, InterPro, KEGG, and AlphaFold/AlphaSync services.
- **High-fidelity structure tracks**: remote GOR4 jobs supply secondary-structure GIFs and residue-by-residue probability tables, backed by AlphaSync and heuristic fallbacks.
- **Topology & localization insights**: automatic Phobius submissions deliver PNG plots, raw text outputs, and transporter heuristics.
- **Network-aware exploration**: STRING Cytoscape exports, Louvain community detection, and GO/KEGG summaries surface pathway-level signals.
- **Interactive web UI**: FastAPI + Jinja templates + modern CSS render run status, filtering, and per-gene drill-downs with inline downloads.
- **Run orchestration**: every request gets its own `runs/<id>` sandbox with cached logs, inputs, and pipeline artifacts for reproducibility.

## Repository Layout
```
webapp/
├── main_final.py           # Offline pipeline / annotator
├── server.py               # FastAPI server + run manager
├── templates/              # Results, downloads, help pages
├── static/style.css        # Custom styling for the web UI
├── requirements.txt        # Pip-based dependency list
├── environment.yml         # Conda environment spec (Python 3.10)
├── runs/                   # Per-run inputs, outputs, caches
└── README.md               # This guide
```

## Prerequisites
- Python 3.10+
- `conda` or `venv` for isolation (recommended)
- Network access to public bioinformatics services (UniProt, Ensembl, STRING, EggNOG, InterPro, KEGG, Phobius, Alphasync, GOR4/NPS@)

## Installation
### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate transcripomics_server
```

### Pip / virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline
1. Place your differential expression file at `GEO_data.csv` (or point `EXPR_FILE` to a custom path).
2. Optionally tweak thresholds via environment variables (see below).
3. Execute:
   ```bash
   python main_final.py
   ```
4. Results land in `pipeline_results/` with caches in `pipeline_results/cache/`, plus per-gene folders under `pipeline_results/genes/{upregulated,downregulated}/<gene_id>/`.
5. Each gene directory contains FASTA, GOR4 GIFs/text, Phobius outputs, STRING exports, and CSV summaries for downstream tooling.

## Serving the Web UI
After generating at least one run, launch the FastAPI app:
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```
- Visit `http://localhost:8000` to upload new expression tables, monitor run logs, and browse results.
- New uploads are stored under `runs/<run_id>/` with
  - `upload.csv` – original file
  - `GEO_data.csv` – working copy for the pipeline
  - `pipeline_results/` – output bundle used by the UI

## Configurable Environment Variables
| Variable | Default | Purpose |
| --- | --- | --- |
| `EXPR_FILE` | `GEO_data.csv` | Input expression matrix for standalone runs |
| `OUTPUT_DIR` | `pipeline_results` | Destination for CSV outputs |
| `CACHE_DIR` | `${OUTPUT_DIR}/cache` | Cache + `.done` markers |
| `GENE_DIR` | `${OUTPUT_DIR}/genes` | Root for per-gene folders |
| `UP_GENES_DIR` / `DOWN_GENES_DIR` | `${GENE_DIR}/upregulated` / `downregulated` | Split gene outputs by regulation |
| `NETWORK_DIR` | `${OUTPUT_DIR}/networks` | STRING network exports |
| `LOG2FC_THRESHOLD` | `1.0` | Absolute log2 fold-change cutoff |
| `PVALUE_THRESHOLD` | `0.05` | Significance filter (raw p-value) |
| `DISTANCE_PERCENTILE` | `95` | Graph-pruning percentile for network visualization |
| `USE_ADJUSTED_PVALUE` | `False` | Switch to FDR filtering |
| `RUN_DASH_SERVER` | `True` | Keeps Dash callbacks enabled if embedding elsewhere |
| `PHOBIUS_EMAIL` | `wp-angular-web@ebi.ac.uk` | Email required by the Phobius REST API |
| `GOR4_ALI_WIDTH` | `70` | Alignment width passed to GOR4 |
| `GOR4_MAX_POLLS` | `20` | Number of polling attempts for GOR4 artifacts |
| `GOR4_POLL_DELAY` | `2.0` seconds | Delay between polls |

> Tip: export variables before launching either `main_final.py` or `uvicorn` to keep configuration consistent across the pipeline and server.

## External Services & Artifacts
- **UniProt / Ensembl / STRING / KEGG / EggNOG / InterPro** for annotation, GO terms, and pathways.
- **AlphaSync** for remote secondary-structure calls; **GOR4** for high-quality GIFs & tables; internal heuristics ensure a fallback track.
- **Phobius** for TM/topology predictions with PNG and text payloads.
- **Plotly, Dash, Cytoscape** for interactive charts and networks rendered in-browser.

## Typical Workflow
1. Upload or place a differential expression CSV (`gene_id`, `log2FC`, `p_value` required).
2. Kick off a run via the UI (or execute `python main_final.py`).
3. Wait for remote API calls (GOR4, Phobius, STRING) to finish; progress is logged under `runs/<id>/pipeline.log`.
4. Explore results through the Secondary Structure, TM Helix, Orthologs, GO, KEGG, and Network tabs.
5. Download per-gene bundles or aggregated CSVs from the Downloads panel for publication-ready figures.

## Troubleshooting
- **External API timeouts**: re-run the pipeline; cached responses live under `pipeline_results/cache/` to avoid repeated calls.
- **GOR4 capacity errors**: adjust `GOR4_MAX_POLLS`/`GOR4_POLL_DELAY` or retry when the public server is less busy.
- **Phobius quota**: ensure `PHOBIUS_EMAIL` uses a valid address per EMBL-EBI policy.
- **Dash assets not loading**: confirm `uvicorn` is serving `/static` correctly and no reverse proxy is stripping the path.

## Contributing
1. Fork / branch.
2. Add or update automated tests/notebooks as needed.
3. Run `python main_final.py` on a sample CSV plus `uvicorn server:app --reload` to validate both CLI and web behavior.
4. Submit a PR with a concise summary of the change and any new environment knobs.

---
Need help? Open an issue describing the dataset, configuration, and full stack trace/log excerpt so we can reproduce quickly.
