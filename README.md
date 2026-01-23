# Transcriptomics Annotation Pipeline Web Server

This project powers a hosted transcriptomics portal that anyone can visit through a standard browser to upload GEO-style differential expression files, trigger automated annotation, and explore beautifully rendered gene dossiers. No local installation is required—the service orchestrates every downstream API call (UniProt, Ensembl, STRING, KEGG, EggNOG, InterPro, Phobius, GOR4, AlphaSync) and returns curated visualizations, tables, and download bundles.

## Highlights at a Glance
- **End-to-end automation** – From CSV ingestion to final visualization, each run is processed, cached, and versioned under a unique run ID for reproducibility.
- **Deep annotation stack** – GO terms, pathways, domain families, STRING networks, and ortholog groups are fetched in the background and summarized in the UI.
- **Premium structure insights** – Secondary-structure GIFs plus residue-wise probability tables (GOR4) sit alongside AlphaSync results and heuristic backups, so every gene receives a structural narrative.
- **Transmembrane awareness** – Phobius submissions yield PNG plots, raw text, and transporter calls that surface localization hints instantly.
- **Download-friendly** – Every panel offers links to the underlying FASTA, networks, CSVs, and per-gene bundles so you can continue analysis offline.

## Using the Web Experience
1. **Visit the deployment URL** shared by your administrator (e.g., `https://your-domain.example.org`).
2. **Upload** a differential expression table (CSV) containing at least `gene_id`, `log2FC`, and `p_value` columns.
3. **Configure thresholds** using the form controls (log2FC, p-value/FDR, network distance percentile) if you want to deviate from defaults.
4. **Submit the run** and leave the tab open; the dashboard streams pipeline logs so you can monitor progress in real time.
5. **Review results** once the status flips to *Completed*. Drill into any gene to see sequence tracks, GOR4 imagery, TM plots, GO terms, pathways, and STRING neighborhoods.
6. **Share or download**: copy the run-specific URL to share with collaborators, or jump to the Downloads page to collect CSV exports and per-gene assets.

## Data Requirements
- CSV (comma-separated) with headers.
- Required columns: `gene_id`, `log2FC`, `p_value` (additional metadata columns are preserved and shown in tables).
- Expression values should already be summarized per gene; log2 fold change can be positive (up) or negative (down).
- P-values must be greater than 0 and preferably already corrected if you plan to enable FDR filtering.

## What You Receive Per Gene
- Cleaned protein FASTA sequence (if available) with download link.
- GOR4 secondary-structure state GIF, confidence GIF, and residue table, all viewable in-browser.
- AlphaSync secondary-structure fallback plus heuristic predictions when remote services cannot resolve the sequence.
- Phobius transmembrane plots, topology text, and transporter heuristic flag.
- STRING interaction subnetwork with community colors and edge scores.
- GO Biological Process / Molecular Function / Cellular Component summaries, InterPro domains, Pfam/CATH hits, KEGG pathways, and EggNOG annotations.
- Orthologous group listing for quick comparative context.

## Output Download Bundles
- **Annotations CSV** – Consolidated metadata for every retained gene.
- **Up/down regulated lists** – Filtered gene tables ready for downstream plotting.
- **GO / KEGG / EggNOG / InterPro exports** – Topic-specific CSVs for enrichment workflows.
- **Per-gene folders** – Contain FASTA, GOR4 artifacts, Phobius files, STRING edge lists, secondary-structure text, and visualization-ready SVG/PNG assets.

## Behind the Scenes
- The processing engine (`main_final.py`) keeps deterministic caches so repeated runs with the same inputs return instantly where possible.
- Remote calls respect public API limits with retry logic, job polling, and graceful fallbacks (e.g., AlphaSync to heuristic secondary structures).
- Every run lives under `/runs/<run_id>/` with its own metadata (`meta.json`), uploaded file, log, and `pipeline_results/` directory, enabling quick auditing or re-downloads.

## Getting Help
- **Need access?** Contact your system administrator for the public URL or credentials if the portal is gated.
- **Run failing?** Use the built-in log viewer linked from each run card to capture stack traces before reaching out.
- **Feature ideas?** Please open an issue or send feedback describing the dataset, objective, and any visual mockups you'd like to see.

Enjoy exploring your transcriptomics data without worrying about local environment setup—the server handles everything, and you only need a browser.
