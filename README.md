# Transcriptomics Annotation Pipeline

This repository powers a production-grade transcriptomics web portal designed for browser-first use. Researchers upload GEO-style differential expression files, launch automated annotation, and explore richly visualized gene dossiers without installing local software. The platform orchestrates downstream services (UniProt, Ensembl, STRING, KEGG, EggNOG, InterPro, Phobius, GOR4, AlphaSync) and delivers curated dashboards, tables, and export-ready bundles.

## Executive Summary
- **Hosted, browser-only workflow** with no local installs.
- **Automated multi-source annotation** across pathways, domains, networks, and orthologs.
- **Run-level provenance** with reproducible outputs and shareable links.
- **Downloadable exports** for downstream analysis pipelines.

## Core Capabilities
- **End-to-end processing** from CSV ingestion to final visualization.
- **Gene-level dossiers** with FASTA, structure, topology, pathways, GO terms, and networks.
- **Interactive exploration** of STRING neighborhoods, KEGG pathways, and domain families.
- **Resilient annotation** via retries, polling, and heuristic fallbacks.
- **Versioned runs** with unique identifiers and cached results for rapid reloads.

## Web Workflow (Hosted)
1. **Open the portal** using the URL provided by your administrator.
2. **Upload** a differential expression CSV with required columns.
3. **Adjust thresholds** (log2FC, p-value/FDR, network distance percentile) if desired.
4. **Start the run** and monitor real-time logs in the browser.
5. **Explore results** once processing completes.
6. **Share or export** results via run links and the Downloads area.

## Input Specifications
**Required columns**
- `gene_id`
- `log2FC`
- `p_value`

**Recommendations**
- Use gene-level summary statistics (not transcript-level).
- Provide corrected p-values if FDR filtering will be applied.
- Additional columns are preserved and shown in results tables.

## Output Artifacts
**Global exports**
- Consolidated annotations CSV
- Upregulated and downregulated gene lists
- GO, KEGG, EggNOG, and InterPro summaries

**Per-gene dossier**
- Protein FASTA (when available)
- Secondary structure: GOR4 visuals + residue table
- AlphaSync results and heuristic backups
- Phobius topology PNG and text
- STRING interaction subnetwork
- GO, KEGG, InterPro, Pfam/CATH, EggNOG annotations

## Reliability and Reproducibility
- Deterministic caching accelerates repeat analyses.
- External API calls honor rate limits with backoff and retries.
- Each run is versioned with metadata, logs, and outputs for auditing.

## Access and Support
- **Access**: Contact your administrator for the hosted URL and credentials.
- **Troubleshooting**: Use the built-in run log viewer to capture errors before contacting support.
- **Feedback**: Open an issue with dataset details and desired enhancements.

Deliver transcriptomics insights through a professional, hosted web platform—no local setup required.
