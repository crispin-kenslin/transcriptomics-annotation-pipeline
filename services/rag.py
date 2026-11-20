from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import httpx
import pandas as pd

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

DB_LINKS = {
    "uniprot": "https://www.uniprot.org/uniprotkb/{id}",
    "kegg_pathway": "https://www.genome.jp/pathway/{id}",
    "interpro": "https://www.ebi.ac.uk/interpro/entry/InterPro/{id}/",
    "pfam": "https://www.ebi.ac.uk/interpro/entry/pfam/{id}/",
    "eggnog": "http://eggnog5.embl.de/#/app/results?target_nogs={id}",
}


@dataclass
class RagSource:
    key: str
    title: str
    url: str
    snippet: str
    source: str


def _clean(val: Any) -> str:
    if val is None:
        return ""
    try:
        if isinstance(val, float) and val != val:
            return ""
    except Exception:
        pass
    text = str(val).strip()
    if text.lower() in ("nan", "none"):
        return ""
    return text


def _split_ids(text: str) -> list[str]:
    if not text:
        return []
    return [t.strip() for t in re.split(r"[;,]", text) if t.strip()]


def extract_genes_from_question(question: str, gene_ids: list[str]) -> list[str]:
    q_upper = question.upper()
    found = [gid for gid in gene_ids if gid and gid.upper() in q_upper]
    return found[:20]


def select_gene_rows(question: str, ann_df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if ann_df.empty or "gene_id" not in ann_df.columns:
        return ann_df

    gene_ids = ann_df["gene_id"].astype(str).tolist()
    mentioned = extract_genes_from_question(question, gene_ids)
    if mentioned:
        return ann_df[ann_df["gene_id"].astype(str).isin(mentioned)].head(max(limit * 2, len(mentioned)))

    keyword_cols = [
        c for c in [
            "protein_name", "function", "GO_biological_process",
            "GO_molecular_function", "GO_cellular_component",
            "InterPro_domains", "Pfam_domains",
        ]
        if c in ann_df.columns
    ]
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", question.lower())]
    if keyword_cols and words:
        mask = pd.Series(False, index=ann_df.index)
        for word in words:
            for col in keyword_cols:
                mask = mask | ann_df[col].astype(str).str.lower().str.contains(re.escape(word), na=False)
        matched = ann_df[mask]
        if not matched.empty:
            return matched.head(limit)

    if "log2FC" in ann_df.columns:
        order = ann_df["log2FC"].abs().sort_values(ascending=False).index
        return ann_df.reindex(order).head(limit)

    return ann_df.head(limit)


def build_gene_context_block(row: pd.Series, kegg_map: dict[str, str], direction: str) -> tuple[str, list[RagSource]]:
    gene_id = _clean(row.get("gene_id"))
    sources: list[RagSource] = []
    lines = [f"### Gene: {gene_id} ({direction})"]

    logfc = _clean(row.get("log2FC"))
    pval = _clean(row.get("p_value"))
    if logfc or pval:
        lines.append(f"log2FC: {logfc or 'NA'} | p-value: {pval or 'NA'}")

    uniprot_id = _clean(row.get("uniprot_id") or row.get("acc"))
    gene_name = _clean(row.get("gene_name"))
    organism = _clean(row.get("organism"))
    protein_name = _clean(row.get("protein_name"))
    function = _clean(row.get("function"))
    if uniprot_id:
        url = DB_LINKS["uniprot"].format(id=uniprot_id)
        lines.append(
            f"[UniProt:{uniprot_id}] Gene: {gene_name or 'Unknown'}. Organism: {organism or 'Unknown'}. "
            f"Protein: {protein_name or 'Unknown'}. Function: {function or 'Not available'}."
        )
        sources.append(RagSource(
            key=f"UniProt:{uniprot_id}", title=f"UniProt {uniprot_id}", url=url,
            snippet=(function or protein_name)[:200], source="UniProt",
        ))

    panther = _clean(row.get("PANTHER"))
    keywords = _clean(row.get("keywords"))
    mol_weight = _clean(row.get("molecular_weight"))
    if panther:
        lines.append(f"[PANTHER] family: {panther}")
    if keywords:
        lines.append(f"[UniProt] keywords: {keywords}")
    if mol_weight:
        lines.append(f"[UniProt] molecular weight: {mol_weight} Da")

    for label, col in [
        ("GO biological process", "GO_biological_process"),
        ("GO molecular function", "GO_molecular_function"),
        ("GO cellular component", "GO_cellular_component"),
    ]:
        val = _clean(row.get(col))
        if val:
            lines.append(f"[GO] {label}: {val}")

    interpro = _clean(row.get("InterPro_domains"))
    if interpro:
        lines.append(f"[InterPro] domains: {interpro}")
        for ipr in _split_ids(interpro)[:3]:
            sources.append(RagSource(
                key=f"InterPro:{ipr}", title=f"InterPro {ipr}",
                url=DB_LINKS["interpro"].format(id=ipr), snippet=interpro[:200], source="InterPro",
            ))

    pfam = _clean(row.get("Pfam_domains"))
    if pfam:
        lines.append(f"[Pfam] domains: {pfam}")
        for pf in _split_ids(pfam)[:3]:
            sources.append(RagSource(
                key=f"Pfam:{pf}", title=f"Pfam {pf}",
                url=DB_LINKS["pfam"].format(id=pf), snippet=pfam[:200], source="Pfam",
            ))

    kegg_text = kegg_map.get(gene_id, "")
    if kegg_text:
        lines.append(f"[KEGG] pathways: {kegg_text}")
        for entry in kegg_text.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            pathway_id = entry.split("|")[0].strip()
            if pathway_id:
                sources.append(RagSource(
                    key=f"KEGG:{pathway_id}", title=f"KEGG {pathway_id}",
                    url=DB_LINKS["kegg_pathway"].format(id=pathway_id), snippet=entry[:200], source="KEGG",
                ))

    orthologs = _clean(row.get("orthologous_groups") or row.get("eggnog_id"))
    if orthologs:
        lines.append(f"[EggNOG] orthologous groups: {orthologs}")

    secondary = _clean(row.get("secondary_structure"))
    if secondary:
        lines.append(f"[GOR4] secondary structure states: {secondary[:150]}")

    tmhmm_text = _clean(row.get("tmhmm_text") or row.get("tmhmm_segments"))
    if tmhmm_text:
        lines.append(f"[TMHMM] topology: {tmhmm_text[:250]}")

    return "\n".join(lines), sources


def build_dataset_context(
    ann_df: pd.DataFrame,
    kegg_df: pd.DataFrame,
    up_df: pd.DataFrame,
    down_df: pd.DataFrame,
    question: str,
    limit: int = 8,
) -> tuple[str, list[RagSource]]:
    if ann_df.empty:
        return "", []

    kegg_map: dict[str, str] = {}
    if not kegg_df.empty and "gene_id" in kegg_df.columns:
        for gid, sub in kegg_df.groupby("gene_id"):
            ids = set()
            for _, r in sub.iterrows():
                kid = _clean(r.get("KEGG_pathway_id")) or _clean(r.get("KEGG_gene_id"))
                if kid:
                    ids.add(kid)
            kegg_map[str(gid)] = "; ".join(sorted(ids))

    up_set = {str(g) for g in up_df.get("gene_id", pd.Series(dtype=str))}
    down_set = {str(g) for g in down_df.get("gene_id", pd.Series(dtype=str))}

    selected = select_gene_rows(question, ann_df, limit=limit)

    blocks: list[str] = []
    all_sources: list[RagSource] = []
    for _, row in selected.iterrows():
        gene_id = _clean(row.get("gene_id"))
        direction = "upregulated" if gene_id in up_set else ("downregulated" if gene_id in down_set else "unchanged")
        block, srcs = build_gene_context_block(row, kegg_map, direction)
        blocks.append(block)
        all_sources.extend(srcs)

    dedup: dict[str, RagSource] = {}
    for s in all_sources:
        dedup.setdefault(s.key, s)

    return "\n\n".join(blocks), list(dedup.values())[:15]


SYSTEM_INSTRUCTIONS = (
    "You are a bioinformatics assistant named 'TAP-bot' answering questions about one completed "
    "transcriptomics annotation run. Answer using the annotation data given "
    "below, which comes directly from this run's own annotations.csv and related "
    "output files (UniProt, KEGG, GO, GOR4 secondary structure, TMHMM topology). "
    "Do not invent gene names, accessions, or facts that are not present in the "
    "data below. If the data below does not contain the answer, say so plainly "
    "instead of guessing. Answer naturally and conversationally."
)


def build_messages(question: str, context_text: str, conversation_history: list[dict[str, str]]) -> list[dict[str, str]]:
    history_text = ""
    for turn in conversation_history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        history_text += f"{role}: {turn.get('content', '')}\n"

    context_block = context_text or "No matching gene data was found in this run for this question."
    user_content = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- ANNOTATION DATA ---\n{context_block}\n--- END ANNOTATION DATA ---\n\n"
        + (f"Conversation so far:\n{history_text}\n" if history_text else "")
        + f"Question: {question}"
    )
    return [{"role": "user", "content": user_content}]


def build_fallback_answer(question: str, context_text: str, sources: list[RagSource]) -> str:
    if not context_text:
        return (
            "I couldn't find any genes or annotation fields in this run's data "
            "that match your question. Try asking about a specific gene ID, "
            "pathway, or protein name from this dataset."
        )
    return (
        "I'm having trouble reaching the language model right now, so here is "
        "the relevant annotation data directly:\n\n" + context_text
    )


async def stream_gemini(messages: list[dict[str, str]], model_options: dict[str, Any] | None = None) -> AsyncGenerator[str, None]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)

    if not api_key:
        yield (
            "Gemini is not configured: no GEMINI_API_KEY was found in this process's "
            "environment. If you set it in a .env file, make sure something actually "
            "loads that file before the server starts (a bare .env file is not read "
            "automatically), then restart the server."
        )
        return

    model_options = model_options or {}
    contents = [{"role": "user", "parts": [{"text": m["content"]}]} for m in messages]
    generation_config = {
        "temperature": model_options.get("temperature", 0.1),
        "topP": model_options.get("top_p", 0.9),
        "maxOutputTokens": model_options.get("max_output_tokens", 2048),
    }
    url = f"{GEMINI_API_BASE}/models/{model}:streamGenerateContent"
    params = {"key": api_key, "alt": "sse"}
    payload = {"contents": contents, "generationConfig": generation_config}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, params=params, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f"Gemini request failed ({resp.status_code}): {body.decode(errors='ignore')[:300]}"
                    return
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data_str)
                    except Exception:
                        continue
                    candidates = parsed.get("candidates") or []
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        text = part.get("text")
                        if text:
                            yield text
    except Exception as exc:
        yield f"Gemini request error: {exc}"
