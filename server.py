import os
import sys
import uuid
import json
import tempfile
import shutil
import subprocess
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Optional, Tuple
import signal

import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np

from plant_registry import (
    DEFAULT_SPECIES_KEY,
    detect_species_from_genes,
    get_species_config,
    list_species_options,
)

ROOT_DIR = Path(__file__).parent
RUNS_DIR = ROOT_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)
PHOBIUS_BASE = "https://www.ebi.ac.uk/Tools/services/rest/phobius"
PHOBIUS_EMAIL = os.getenv("PHOBIUS_EMAIL", "admin@tapipe.res.in")
PHOBIUS_MAX_WAIT = float(os.getenv("PHOBIUS_MAX_WAIT", "45"))
PHOBIUS_POLL_INTERVAL = float(os.getenv("PHOBIUS_POLL_INTERVAL", "0.5"))
PHOBIUS_STATUS_TIMEOUT = float(os.getenv("PHOBIUS_STATUS_TIMEOUT", "10"))
PHOBIUS_RESULT_TIMEOUT = float(os.getenv("PHOBIUS_RESULT_TIMEOUT", "10"))
PHOBIUS_RESULT_RETRIES = int(os.getenv("PHOBIUS_RESULT_RETRIES", "40"))
PHOBIUS_RESULT_DELAY = float(os.getenv("PHOBIUS_RESULT_DELAY", "0.5"))
FAVICON_PATH = ROOT_DIR / "favicon.ico"

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="Transcriptomics Pipeline Web Server", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if FAVICON_PATH.exists():
        return FileResponse(str(FAVICON_PATH))
    raise HTTPException(status_code=404, detail="Favicon not found")


class RunManager:
    def __init__(self):
        self.runs = {}
        self.lock = Lock()
        self.procs = {}

    def _meta_path(self, run_id: str) -> Path:
        return RUNS_DIR / run_id / "meta.json"

    def _save_meta(self, info: dict):
        try:
            meta_path = self._meta_path(info["id"])
            meta = {
                "id": info["id"],
                "name": info.get("name"),
                "status": info.get("status"),
                "created_at": info.get("created_at").isoformat() if info.get("created_at") else None,
                "updated_at": info.get("updated_at").isoformat() if info.get("updated_at") else None,
                "params": info.get("params", {}),
                "summary": info.get("summary", {}),
                "returncode": info.get("returncode"),
                "pid": info.get("pid"),
            }
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def _load_from_disk(self, run_id: str):
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            return None

        meta_path = self._meta_path(run_id)
        params_default = {
            "gene_col": "gene_id",
            "log2fc_col": "log2FC",
            "pval_col": "p_value",
            "adj_p_col": None,
            "log2fc_threshold": 1.0,
            "pvalue_threshold": 0.05,
            "distance_percentile": 95,
            "use_fdr": True,
        }
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

        created_at = datetime.fromisoformat(meta.get("created_at")) if meta.get("created_at") else datetime.utcnow()
        updated_at = datetime.fromisoformat(meta.get("updated_at")) if meta.get("updated_at") else created_at
        params = meta.get("params") or params_default
        name = meta.get("name") or (params.get("job_name") if isinstance(params, dict) else None)

        output_dir = run_dir / "pipeline_results"
        summary = meta.get("summary") or self._collect_summary(output_dir)
        status = meta.get("status") or ("completed" if output_dir.exists() else "unknown")

        info = {
            "id": run_id,
            "name": name,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "run_dir": run_dir,
            "output_dir": output_dir,
            "input_path": run_dir / "upload.csv",
            "expr_path": run_dir / "GEO_data.csv",
            "log_path": run_dir / "pipeline.log",
            "logs": deque(maxlen=800),
            "params": params,
            "summary": summary,
            "returncode": meta.get("returncode"),
            "pid": meta.get("pid"),
        }

        with self.lock:
            self.runs[run_id] = info
        return info

    def _load_all_from_disk(self):
        for run_dir in RUNS_DIR.iterdir():
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            if run_id in self.runs:
                continue
            self._load_from_disk(run_id)

    def list_runs(self):
        self._load_all_from_disk()
        with self.lock:
            # prune entries whose run directory was deleted
            to_delete = []
            for rid, info in self.runs.items():
                run_dir = info.get("run_dir", Path())
                out_dir = info.get("output_dir", Path())
                if not run_dir.exists() or not out_dir.exists():
                    to_delete.append(rid)
            for rid in to_delete:
                self.runs.pop(rid, None)
            return sorted(self.runs.values(), key=lambda r: r["created_at"], reverse=True)

    def update_run_params(self, run_id: str, updates: dict):
        with self.lock:
            info = self.runs.get(run_id)
            if not info:
                raise ValueError("Unknown run")
            info["params"].update(updates)
            info["updated_at"] = datetime.utcnow()
        self._save_meta(info)
        return info

    def get(self, run_id):
        with self.lock:
            found = self.runs.get(run_id)
        if found:
            return found
        return self._load_from_disk(run_id)

    def create(self, params):
        run_id = uuid.uuid4().hex[:8]
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        info = {
            "id": run_id,
            "name": params.get("job_name") or None,
            "status": "queued",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "run_dir": run_dir,
            "output_dir": run_dir / "pipeline_results",
            "input_path": run_dir / "upload.csv",
            "expr_path": run_dir / "GEO_data.csv",
            "log_path": run_dir / "pipeline.log",
            "logs": deque(maxlen=800),
            "params": params,
            "summary": {},
            "returncode": None,
            "pid": None,
        }

        with self.lock:
            self.runs[run_id] = info

        self._save_meta(info)

        return info

    def _collect_summary(self, output_dir: Path):
        summary = {}

        def _count(path):
            if path.exists() and path.stat().st_size > 0:
                try:
                    return len(pd.read_csv(path))
                except Exception:
                    return 0
            return 0

        summary["upregulated_count"] = _count(output_dir / "upregulated_genes.csv")
        summary["downregulated_count"] = _count(output_dir / "downregulated_genes.csv")
        summary["annotation_count"] = _count(output_dir / "annotations.csv")
        summary["kegg_count"] = _count(output_dir / "kegg_pathways.csv")
        summary["eggnog_go_count"] = _count(output_dir / "eggnog_go_terms.csv")
        summary["interpro_count"] = _count(output_dir / "interpro_domains.csv")
        processed = output_dir / "cache" / "processed_data.csv"
        summary["total_genes"] = _count(processed)
        return summary

    def start(self, run_id):
        info = self.get(run_id)
        if not info:
            raise ValueError("Unknown run")

        def _worker():
            with self.lock:
                info["status"] = "running"
                info["updated_at"] = datetime.utcnow()
            self._save_meta(info)

            env = os.environ.copy()
            env.update(
                {
                    "EXPR_FILE": str(info["expr_path"]),
                    "OUTPUT_DIR": str(info["output_dir"]),
                    "LOG2FC_THRESHOLD": str(info["params"]["log2fc_threshold"]),
                    "PVALUE_THRESHOLD": str(info["params"]["pvalue_threshold"]),
                    "DISTANCE_PERCENTILE": str(info["params"]["distance_percentile"]),
                    "USE_ADJUSTED_PVALUE": "1" if info["params"].get("use_fdr") else "0",
                    "RUN_DASH_SERVER": "0",
                }
            )

            species_key = info["params"].get("species_key", DEFAULT_SPECIES_KEY)
            env["PLANT_SPECIES_KEY"] = species_key
            env["PLANT_SPECIES_LABEL"] = info["params"].get(
                "species_label", species_key.title()
            )
            env["PLANT_SPECIES_SHORT_NAME"] = info["params"].get(
                "species_short_name", env["PLANT_SPECIES_LABEL"]
            )
            env["SPECIES_SCIENTIFIC_NAME"] = info["params"].get(
                "species_scientific_name", env["PLANT_SPECIES_LABEL"]
            )
            env["UNIPROT_TAX_ID"] = str(info["params"].get("uniprot_tax_id", 39947))
            env["STRING_SPECIES"] = str(info["params"].get("string_species_id", 39947))

            # Assume main_final.py is in the same folder as server.py
            main_py_path = Path(__file__).parent / "main_final.py"
            cmd = [
                sys.executable,
                "-u",
                str(main_py_path),
            ]

            with info["log_path"].open("a", encoding="utf-8") as log_file:
                log_file.write(f"Run started at {datetime.utcnow().isoformat()} UTC\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Environment overrides: OUTPUT_DIR={info['output_dir']}\n")
                log_file.flush()

                try:
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(ROOT_DIR),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    with self.lock:
                        info["pid"] = process.pid
                        self.procs[run_id] = process
                    self._save_meta(info)
                except Exception as exc:  # pragma: no cover
                    with self.lock:
                        info["status"] = "failed"
                        info["updated_at"] = datetime.utcnow()
                        info["logs"].append(f"Failed to start pipeline: {exc}")
                    log_file.write(f"Failed to start pipeline: {exc}\n")
                    return

                for line in process.stdout:  # type: ignore[attr-defined]
                    clean = line.rstrip("\n")
                    info["logs"].append(clean)
                    log_file.write(line)
                    log_file.flush()

                process.wait()
                info["returncode"] = process.returncode

                with self.lock:
                    existing_status = info.get("status")
                    info["status"] = (
                        "stopped"
                        if existing_status == "stopped"
                        else ("completed" if process.returncode == 0 else "failed")
                    )
                    info["updated_at"] = datetime.utcnow()
                    if process.returncode == 0:
                        info["summary"] = self._collect_summary(info["output_dir"])
                    self.procs.pop(run_id, None)
                self._save_meta(info)

                log_file.write(f"Run finished with code {process.returncode}\n")

        Thread(target=_worker, daemon=True).start()

    def logs_as_text(self, run_id):
        info = self.get(run_id)
        if not info:
            return ""

        text = "\n".join(info["logs"])
        if info["log_path"].exists():
            try:
                with info["log_path"].open("r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                pass
        return text

    def tail_logs(self, run_id, max_lines=400):
        info = self.get(run_id)
        if not info:
            return []
        logs = list(info["logs"])
        if logs:
            return logs[-max_lines:]
        if info["log_path"].exists():
            try:
                lines = info["log_path"].read_text(encoding="utf-8", errors="ignore").splitlines()
                return lines[-max_lines:]
            except Exception:
                return []
        return []

    def stop(self, run_id):
        info = self.get(run_id)
        if not info:
            raise ValueError("Unknown run")

        with self.lock:
            status = info.get("status")
            pid = info.get("pid")
            process = self.procs.get(run_id)

        if status in {"completed", "failed", "stopped"}:
            return False

        # mark as stopping
        with self.lock:
            info["status"] = "stopped"
            info["updated_at"] = datetime.utcnow()
            info["logs"].append("Run was requested to stop.")
        self._save_meta(info)

        try:
            if process:
                process.terminate()
            elif pid:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            pass

        return True


manager = RunManager()


def _prepare_expression(
    upload_path: Path,
    expr_path: Path,
    gene_col: str,
    log2fc_col: str,
    pval_col: str,
    adj_p_col: Optional[str] = None,
) -> Tuple[int, Optional[str]]:
    try:
        df = pd.read_csv(upload_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {exc}")

    adj_p_col_clean = (adj_p_col or "").strip()
    required_cols = [gene_col, log2fc_col, pval_col]
    if adj_p_col_clean:
        required_cols.append(adj_p_col_clean)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")

    adj_series = None
    if adj_p_col_clean:
        adj_series = df[adj_p_col_clean].copy()

    renamed = df.rename(columns={gene_col: "gene_id", log2fc_col: "log2FC", pval_col: "p_value"})
    renamed["log2FC"] = pd.to_numeric(renamed["log2FC"], errors="coerce")
    renamed["p_value"] = pd.to_numeric(renamed["p_value"], errors="coerce")
    renamed = renamed.dropna(subset=["gene_id", "log2FC", "p_value"])

    if adj_series is not None:
        renamed["adj_p_value"] = pd.to_numeric(adj_series, errors="coerce")

    if renamed.empty:
        raise HTTPException(status_code=400, detail="No valid rows after cleaning.")

    renamed.to_csv(expr_path, index=False)
    inferred = detect_species_from_genes(renamed["gene_id"]) if not renamed.empty else None
    return len(renamed), inferred


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    runs = manager.list_runs()
    for r in runs:
        r["name"] = r.get("name") or (r.get("params", {}) or {}).get("job_name")
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "runs": runs,
            "species_options": list_species_options(),
            "default_species": DEFAULT_SPECIES_KEY,
        },
    )


@app.get("/run", response_class=HTMLResponse)
async def run_launcher(request: Request):
    runs = manager.list_runs()
    for r in runs:
        r["name"] = r.get("name") or (r.get("params", {}) or {}).get("job_name")
    return TEMPLATES.TemplateResponse(
        "run_launcher.html",
        {
            "request": request,
            "runs": runs,
            "species_options": list_species_options(),
            "default_species": DEFAULT_SPECIES_KEY,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    runs = manager.list_runs()
    for r in runs:
        r["name"] = r.get("name") or (r.get("params", {}) or {}).get("job_name")
    return TEMPLATES.TemplateResponse(
        "history.html",
        {
            "request": request,
            "runs": runs,
        },
    )


@app.get("/species", response_class=HTMLResponse)
async def species_page(request: Request):
    return TEMPLATES.TemplateResponse(
        "species.html",
        {
            "request": request,
            "species_options": list_species_options(),
        },
    )


@app.get("/workflow", response_class=HTMLResponse)
async def workflow_page(request: Request):
    runs = manager.list_runs()
    return TEMPLATES.TemplateResponse(
        "workflow.html",
        {
            "request": request,
            "runs": runs,
        },
    )


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    return TEMPLATES.TemplateResponse("help.html", {"request": request})


@app.post("/run")
async def run_pipeline(
    request: Request,
    data_file: UploadFile = File(...),
    gene_col: str = Form("gene_id"),
    log2fc_col: str = Form("log2FC"),
    pval_col: str = Form("p_value"),
    log2fc_threshold: float = Form(1.0),
    pvalue_threshold: float = Form(0.05),
    distance_percentile: float = Form(95),
    use_fdr: bool = Form(False),
    job_name: str = Form(""),
    species_key: str = Form("auto"),
    adj_p_col: str = Form(""),
):
    job_name_clean = job_name.strip() or None
    requested_species = (species_key or "auto").strip().lower() or "auto"
    adj_col_clean = (adj_p_col or "").strip()
    params = {
        "gene_col": gene_col.strip(),
        "log2fc_col": log2fc_col.strip(),
        "pval_col": pval_col.strip(),
        "adj_p_col": adj_col_clean or None,
        "log2fc_threshold": log2fc_threshold,
        "pvalue_threshold": pvalue_threshold,
        "distance_percentile": distance_percentile,
        "use_fdr": use_fdr,
        "job_name": job_name_clean,
        "species_key": requested_species,
        "species_detection": "pending",
    }

    info = manager.create(params)

    with info["input_path"].open("wb") as buffer:
        shutil.copyfileobj(data_file.file, buffer)

    cleaned_rows, inferred_species = _prepare_expression(
        info["input_path"],
        info["expr_path"],
        params["gene_col"],
        params["log2fc_col"],
        params["pval_col"],
        params.get("adj_p_col"),
    )

    final_species_key = requested_species
    detection_label = "user-selected"
    if requested_species == "auto":
        if inferred_species:
            final_species_key = inferred_species
            detection_label = "auto-detected"
        else:
            final_species_key = DEFAULT_SPECIES_KEY
            detection_label = "auto-fallback"

    species_cfg = get_species_config(final_species_key)
    info = manager.update_run_params(
        info["id"],
        {
            "species_key": species_cfg.key,
            "species_label": species_cfg.label,
            "species_scientific_name": species_cfg.scientific_name,
            "species_short_name": species_cfg.short_name,
            "string_species_id": species_cfg.string_species_id,
            "uniprot_tax_id": species_cfg.uniprot_tax_id,
            "species_detection": detection_label,
            "species_auto_candidate": inferred_species,
            "cleaned_rows": cleaned_rows,
            "adj_p_col": params.get("adj_p_col"),
        },
    )

    manager.start(info["id"])

    url = request.url_for("view_run", run_id=info["id"])
    response = RedirectResponse(url=str(url), status_code=303)
    response.set_cookie("last_run", info["id"])
    return response


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    info["name"] = info.get("name") or (info.get("params", {}) or {}).get("job_name")
    return TEMPLATES.TemplateResponse(
        "run.html",
        {"request": request, "run": info, "species_params": info.get("params", {})},
    )


@app.get("/runs/{run_id}/status", response_class=JSONResponse)
async def run_status(run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": run_id,
        "status": info["status"],
        "logs": manager.tail_logs(run_id),
        "summary": info.get("summary", {}),
    }


@app.get("/runs/{run_id}/logs", response_class=PlainTextResponse)
async def stream_logs(run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    return manager.logs_as_text(run_id)


@app.post("/runs/{run_id}/stop", response_class=JSONResponse)
async def stop_run(run_id: str):
    try:
        stopped = manager.stop(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found")
    if not stopped:
        raise HTTPException(status_code=400, detail="Run is not running")
    return {"status": "stopped", "id": run_id}


@app.get("/runs/{run_id}/results", response_class=HTMLResponse)
async def view_results(request: Request, run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run has not completed yet")

    info["name"] = info.get("name") or (info.get("params", {}) or {}).get("job_name")

    output_dir = info["output_dir"]

    def _head(path, n=25):
        if path.exists() and path.stat().st_size > 0:
            try:
                return pd.read_csv(path).head(n)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _table(df):
        if df is None or df.empty:
            return {"cols": [], "rows": []}
        return {"cols": list(df.columns), "rows": df.to_dict(orient="records")}

    up_table = _table(_head(output_dir / "upregulated_genes.csv"))
    down_table = _table(_head(output_dir / "downregulated_genes.csv"))
    ann_table = _table(_head(output_dir / "annotations.csv"))
    kegg_table = _table(_head(output_dir / "kegg_pathways.csv"))

    downloads = []
    for fname in [
        "upregulated_genes.csv",
        "downregulated_genes.csv",
        "annotations.csv",
        "kegg_pathways.csv",
        "eggnog_go_terms.csv",
        "eggnog_domains.csv",
        "interpro_domains.csv",
    ]:
        fpath = output_dir / fname
        if fpath.exists():
            downloads.append({"label": fname, "path": f"/runs/{run_id}/files/{fname}"})

    networks = []
    for name in ["upregulated", "downregulated"]:
        svg_path = output_dir / "networks" / name / f"{name}_network.svg"
        if svg_path.exists():
            networks.append(
                {
                    "label": f"{name.title()} network",
                    "view_url": f"/runs/{run_id}/files/networks/{name}/{name}_network.svg",
                    "download_url": f"/runs/{run_id}/files/networks/{name}/{name}_network.svg",
                }
            )

    summary = info.get("summary", {})
    if not summary or "total_genes" not in summary:
        summary = manager._collect_summary(output_dir)
        info["summary"] = summary
        manager._save_meta(info)

    # Compute runtime from created_at/updated_at if available
    run_duration = None
    try:
        started = info.get("created_at")
        ended = info.get("updated_at")
        if started and ended:
            delta = ended - started
            total_seconds = int(delta.total_seconds())
            mins, secs = divmod(total_seconds, 60)
            hours, mins = divmod(mins, 60)
            if hours:
                run_duration = f"{hours}h {mins}m {secs}s"
            elif mins:
                run_duration = f"{mins}m {secs}s"
            else:
                run_duration = f"{secs}s"
    except Exception:
        run_duration = None

    context = {
        "request": request,
        "run": info,
        "run_job_name": info.get("name") or (info.get("params", {}) or {}).get("job_name"),
        "summary": summary,
        "up_table": up_table,
        "down_table": down_table,
        "ann_table": ann_table,
        "kegg_table": kegg_table,
        "downloads": downloads,
        "networks": networks,
        "run_id": run_id,
        "run_duration": run_duration,
        "species_params": info.get("params", {}),
    }
    return TEMPLATES.TemplateResponse("results.html", context)


@app.get("/runs/{run_id}/files/{file_path:path}")
async def serve_run_file(run_id: str, file_path: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")

    requested = (info["output_dir"] / file_path).resolve()
    try:
        requested.relative_to(info["output_dir"].resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(requested)


@app.get("/runs/{run_id}/download-zip")
@app.get("/runs/{run_id}/download-zip/")
@app.get("/runs/{run_id}/download_zip")
async def download_zip(run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    output_dir = info["output_dir"]
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output not found")

    tmpdir = tempfile.mkdtemp()
    zip_base = Path(tmpdir) / run_id
    try:
        shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create zip: {exc}")

    zip_path = zip_base.with_suffix(".zip")
    return FileResponse(zip_path, media_type="application/zip", filename=f"{run_id}_output.zip")


@app.get("/runs/{run_id}/downloads", response_class=HTMLResponse)
async def view_downloads(request: Request, run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run has not completed yet")

    output_dir = info["output_dir"]

    def _top_counts(df, key_col, name_col=None, category_col=None, n=12):
        if df is None or df.empty or key_col not in df:
            return []
        counts = df[key_col].dropna().value_counts().head(n)
        rows = []
        for key, cnt in counts.items():
            row = {"key": key, "count": int(cnt)}
            if name_col and name_col in df.columns:
                name_val = df.loc[df[key_col] == key, name_col].dropna().astype(str)
                if not name_val.empty:
                    row["name"] = name_val.iloc[0]
            if category_col and category_col in df.columns:
                cat_val = df.loc[df[key_col] == key, category_col].dropna().astype(str)
                if not cat_val.empty:
                    row["category"] = cat_val.iloc[0]
            rows.append(row)
        return rows

    eggnog_go_df = _safe_read_csv(output_dir / "eggnog_go_terms.csv")
    eggnog_dom_df = _safe_read_csv(output_dir / "eggnog_domains.csv")
    interpro_df = _safe_read_csv(output_dir / "interpro_domains.csv")
    annotations_df = _safe_read_csv(output_dir / "annotations.csv")

    go_summary = _top_counts(eggnog_go_df, "GO_ID", name_col="GO_name", category_col="GO_category")
    eggnog_dom_summary = _top_counts(eggnog_dom_df, "domain_name", name_col="domain_type")
    interpro_summary = _top_counts(interpro_df, "domain_name", name_col="domain_type")

    def _split_counts(series, n=12):
        counts = {}
        for val in series.dropna().astype(str):
            for token in re.split(r"[;,]", val):
                token = token.strip()
                if not token:
                    continue
                counts[token] = counts.get(token, 0) + 1
        return [
            {"key": k, "count": v}
            for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
        ]

    if not go_summary and not annotations_df.empty:
        for col in ["GO_biological_process", "GO_molecular_function", "GO_cellular_component"]:
            if col in annotations_df:
                go_summary = _split_counts(annotations_df[col])
                if go_summary:
                    break

    if not interpro_summary and not annotations_df.empty:
        for col in ["InterPro_domains", "interpro_domains"]:
            if col in annotations_df:
                interpro_summary = _split_counts(annotations_df[col])
                if interpro_summary:
                    break

    if not eggnog_dom_summary and not annotations_df.empty:
        for col in ["Pfam_domains", "CATH_domains", "EggNOG_domains", "eggnog_domains"]:
            if col in annotations_df:
                eggnog_dom_summary = _split_counts(annotations_df[col])
                if eggnog_dom_summary:
                    break

    downloads = []
    for fname in [
        "upregulated_genes.csv",
        "downregulated_genes.csv",
        "annotations.csv",
        "kegg_pathways.csv",
        "eggnog_go_terms.csv",
        "eggnog_domains.csv",
        "interpro_domains.csv",
    ]:
        fpath = output_dir / fname
        if fpath.exists():
            downloads.append({"label": fname, "path": f"/runs/{run_id}/files/{fname}"})

    networks = []
    for name in ["upregulated", "downregulated"]:
        svg_path = output_dir / "networks" / name / f"{name}_network.svg"
        if svg_path.exists():
            networks.append(
                {
                    "label": f"{name.title()} network",
                    "view_url": f"/runs/{run_id}/files/networks/{name}/{name}_network.svg",
                    "download_url": f"/runs/{run_id}/files/networks/{name}/{name}_network.svg",
                }
            )

    context = {
        "request": request,
        "run": info,
        "downloads": downloads,
        "networks": networks,
        "go_summary": go_summary,
        "eggnog_dom_summary": eggnog_dom_summary,
        "interpro_summary": interpro_summary,
        "species_params": info.get("params", {}),
    }
    return TEMPLATES.TemplateResponse("downloads.html", context)


@app.get("/runs/open")
async def open_run(run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info["status"] != "completed":
        raise HTTPException(status_code=404, detail="Run has not completed yet")
    return RedirectResponse(url=f"/runs/{run_id}/results", status_code=303)


def _safe_read_csv(path: Path):
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _sf(val):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return None
        return float(val)
    except Exception:
        return None


def _run_esm_fold(seq: str, out_path: Path, timeout: int = 120):
    return None


def _clean_sequence(seq: str):
    return re.sub(r"[^A-Za-z]", "", seq or "").upper()


def _phobius_submit(fasta_str: str):
    resp = requests.post(
        f"{PHOBIUS_BASE}/run",
        data={"sequence": fasta_str, "email": PHOBIUS_EMAIL},
        headers={"User-Agent": "tapipe-phobius-client"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text.strip()


def _phobius_poll(job_id: str, timeout: Optional[float] = None, interval: Optional[float] = None):
    wait_timeout = timeout or PHOBIUS_MAX_WAIT
    poll_interval = interval or PHOBIUS_POLL_INTERVAL
    status_url = f"{PHOBIUS_BASE}/status/{job_id}"
    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        r = requests.get(status_url, timeout=PHOBIUS_STATUS_TIMEOUT)
        r.raise_for_status()
        status = (r.text or "").strip().upper()
        if status == "FINISHED":
            return True
        if status == "ERROR":
            raise RuntimeError("Phobius job failed")
        time.sleep(poll_interval)
    raise TimeoutError("Phobius job timed out")


def _phobius_fetch_artifact(url: str, binary: bool = True, optional: bool = False):
    retries = max(PHOBIUS_RESULT_RETRIES, 1)
    headers = {"User-Agent": "tapipe-phobius-client"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=PHOBIUS_RESULT_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                return resp.content if binary else resp.text
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(PHOBIUS_RESULT_DELAY)
    if optional:
        return None
    raise TimeoutError(f"Phobius artifact not ready: {url}")


def _phobius_download(job_id: str, gene_dir: Path, gene_id: str):
    gene_dir.mkdir(parents=True, exist_ok=True)
    txt_path = gene_dir / f"tmhmm_{gene_id}.txt"
    png_path = gene_dir / f"tmhmm_{gene_id}.png"

    text_url = f"{PHOBIUS_BASE}/result/{job_id}/out"
    png_url = f"{PHOBIUS_BASE}/result/{job_id}/visual-png"
    txt_content = _phobius_fetch_artifact(text_url, binary=True)
    txt_path.write_bytes(txt_content)

    png_content = _phobius_fetch_artifact(png_url, binary=True, optional=True)
    if png_content:
        png_path.write_bytes(png_content)
    else:
        png_path = None

    try:
        txt_content = txt_path.read_text(errors="ignore")
    except Exception:
        txt_content = ""
    return txt_content, txt_path, png_path


def _run_phobius(seq: str, gene_dir: Path, gene_id: str, force: bool = False):
    gene_dir.mkdir(parents=True, exist_ok=True)
    txt_path = gene_dir / f"tmhmm_{gene_id}.txt"
    png_path = gene_dir / f"tmhmm_{gene_id}.png"

    if not force and txt_path.exists() and png_path.exists():
        try:
            cached_text = txt_path.read_text(errors="ignore")
        except Exception:
            cached_text = ""
        return {"text": cached_text, "text_path": txt_path, "png_path": png_path, "cached": True}

    fasta_payload = f">{gene_id}\n{seq}\n"
    job_id = _phobius_submit(fasta_payload)
    _phobius_poll(job_id)
    text, txt_path, png_path = _phobius_download(job_id, gene_dir, gene_id)
    return {"text": text, "text_path": txt_path, "png_path": png_path, "job_id": job_id, "cached": False}


def _clean_json(obj):
    """Recursively replace NaN/Inf with None so json.dumps succeeds."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    try:
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
    except Exception:
        pass
    return obj


@app.get("/runs/{run_id}/volcano", response_class=JSONResponse)
async def volcano_data(run_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run not completed")

    output_dir = info["output_dir"]

    def _clean_str(val):
        try:
            if val is None:
                return ""
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return ""
            return str(val).strip()
        except Exception:
            return ""

    processed = _safe_read_csv(output_dir / "cache" / "processed_data.csv")
    up_df = _safe_read_csv(output_dir / "upregulated_genes.csv")
    down_df = _safe_read_csv(output_dir / "downregulated_genes.csv")
    ann_df = _safe_read_csv(output_dir / "annotations.csv")
    ann_map = {}
    if not ann_df.empty and "gene_id" in ann_df.columns:
        ann_map = ann_df.set_index("gene_id").to_dict(orient="index")

    # Preload pathway/domain/GO mappings for richer search index
    kegg_df = _safe_read_csv(output_dir / "kegg_pathways.csv")
    kegg_map = {}
    if not kegg_df.empty and "gene_id" in kegg_df.columns:
        for gid, sub in kegg_df.groupby("gene_id"):
            entries = []
            for _, r in sub.iterrows():
                ko = _clean_str(r.get("KO"))
                pathway = _clean_str(r.get("KEGG_pathway_id")) or _clean_str(r.get("KEGG_pathway_name"))
                parts = [p for p in [ko, pathway] if p]
                if parts:
                    entries.append(" | ".join(parts))
            kegg_map[str(gid)] = "; ".join(sorted(set(entries)))

    eggnog_go_df = _safe_read_csv(output_dir / "eggnog_go_terms.csv")
    eggnog_go_map = {}
    if not eggnog_go_df.empty and "gene_id" in eggnog_go_df.columns:
        for gid, sub in eggnog_go_df.groupby("gene_id"):
            entries = []
            for _, r in sub.iterrows():
                cat = _clean_str(r.get("GO_category"))
                go_id = _clean_str(r.get("GO_ID"))
                name = _clean_str(r.get("GO_name"))
                entry = " ".join([p for p in [cat, go_id, name] if p]).strip()
                if entry:
                    entries.append(entry)
            eggnog_go_map[str(gid)] = "; ".join(sorted(set(entries)))

    eggnog_dom_df = _safe_read_csv(output_dir / "eggnog_domains.csv")
    eggnog_dom_map = {}
    if not eggnog_dom_df.empty and "gene_id" in eggnog_dom_df.columns:
        for gid, sub in eggnog_dom_df.groupby("gene_id"):
            entries = []
            for _, r in sub.iterrows():
                name = _clean_str(r.get("domain_name"))
                dtype = _clean_str(r.get("domain_type"))
                entry = " ".join([p for p in [name, dtype] if p]).strip()
                if entry:
                    entries.append(entry)
            eggnog_dom_map[str(gid)] = "; ".join(sorted(set(entries)))

    interpro_df = _safe_read_csv(output_dir / "interpro_domains.csv")
    interpro_map = {}
    if not interpro_df.empty and "gene_id" in interpro_df.columns:
        for gid, sub in interpro_df.groupby("gene_id"):
            entries = []
            for _, r in sub.iterrows():
                iid = _clean_str(r.get("interpro_id"))
                name = _clean_str(r.get("domain_name"))
                entry = " ".join([p for p in [iid, name] if p]).strip()
                if entry:
                    entries.append(entry)
            interpro_map[str(gid)] = "; ".join(sorted(set(entries)))

    if processed.empty:
        return {"points": []}

    up_set = set(up_df.get("gene_id", []))
    down_set = set(down_df.get("gene_id", []))

    points = []
    for _, row in processed.iterrows():
        gene_id = str(row.get("gene_id", ""))
        if not gene_id:
            continue
        log2fc = row.get("log2FC", np.nan)
        pval = row.get("p_value", np.nan)
        adj = row.get("adj_p_value", np.nan)
        neglog = row.get("neg_log10_p", np.nan)
        label = "non";
        if gene_id in up_set:
            label = "up"
        elif gene_id in down_set:
            label = "down"

        ann = ann_map.get(gene_id, {})
        protein = _clean_str(ann.get("protein_name", "Unknown")) if isinstance(ann, dict) else "Unknown"
        func = _clean_str(ann.get("function", "Unknown")) if isinstance(ann, dict) else "Unknown"
        go_bp = _clean_str(ann.get("GO_biological_process", "")) if isinstance(ann, dict) else ""
        go_mf = _clean_str(ann.get("GO_molecular_function", "")) if isinstance(ann, dict) else ""
        go_cc = _clean_str(ann.get("GO_cellular_component", "")) if isinstance(ann, dict) else ""
        pfam = _clean_str(ann.get("Pfam_domains", "")) if isinstance(ann, dict) else ""
        interpro_dom = _clean_str(ann.get("InterPro_domains", "")) if isinstance(ann, dict) else ""
        cath = _clean_str(ann.get("CATH_domains", "")) if isinstance(ann, dict) else ""
        orthologs = _clean_str(ann.get("orthologous_groups", "")) if isinstance(ann, dict) else ""
        domains = []
        if isinstance(ann, dict):
            for key in ["InterPro_domains", "Pfam_domains", "CATH_domains"]:
                val = ann.get(key)
                if isinstance(val, str) and val:
                    domains.append(val)
        domain_text = " | ".join(domains)

        points.append({
            "gene_id": gene_id,
            "log2FC": _sf(log2fc),
            "p_value": _sf(pval),
            "adj_p_value": _sf(adj),
            "neg_log10_p": _sf(neglog),
            "label": label,
            "protein": protein or "Unknown",
            "function": func or "Unknown",
            "domains": domain_text,
            "go_bp": go_bp,
            "go_mf": go_mf,
            "go_cc": go_cc,
            "pfam_domains": pfam,
            "interpro_domains": interpro_dom,
            "cath_domains": cath,
            "orthologs": orthologs,
            "kegg_text": kegg_map.get(gene_id, ""),
            "eggnog_go_text": eggnog_go_map.get(gene_id, ""),
            "eggnog_domain_text": eggnog_dom_map.get(gene_id, ""),
            "interpro_text": interpro_map.get(gene_id, ""),
        })

    return {"points": points}


@app.get("/runs/{run_id}/gene/{gene_id}", response_class=JSONResponse)
async def gene_detail(run_id: str, gene_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run not completed")

    output_dir = info["output_dir"]
    ann_df = _safe_read_csv(output_dir / "annotations.csv")
    if ann_df.empty or "gene_id" not in ann_df.columns:
        raise HTTPException(status_code=404, detail="No annotation data")

    row = ann_df.loc[ann_df["gene_id"] == gene_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Gene not found")
    row = row.iloc[0]

    log2fc = row.get("log2FC", 0)
    regulation = "up" if pd.notna(log2fc) and log2fc > 0 else "down"
    gene_dir = output_dir / "genes" / ("upregulated" if regulation == "up" else "downregulated") / gene_id

    fasta = ""
    fasta_path = gene_dir / f"protein_{gene_id}.fasta"
    if fasta_path.exists():
        try:
            fasta = fasta_path.read_text()
        except Exception:
            fasta = ""
    plain_sequence = ""
    if fasta:
        try:
            plain_sequence = "".join([
                ln.strip() for ln in fasta.splitlines() if ln.strip() and not ln.startswith(">")
            ])
        except Exception:
            plain_sequence = ""

    pdb_text = ""
    pdb_path = gene_dir / f"structure_{gene_id}.pdb"
    if pdb_path.exists():
        try:
            pdb_text = pdb_path.read_text()
        except Exception:
            pdb_text = ""

    svg_path = gene_dir / f"string_{gene_id}.svg"
    svg_url = f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/string_{gene_id}.svg" if svg_path.exists() else None

    edges_path = gene_dir / f"string_edges_{gene_id}.csv"
    edges = []
    if edges_path.exists():
        try:
            df_edges = pd.read_csv(edges_path)
            for _, erow in df_edges.iterrows():
                edges.append({
                    "source": erow.get("source"),
                    "target": erow.get("target"),
                    "combined_score": float(erow.get("combined_score", 0)) if not pd.isna(erow.get("combined_score")) else None,
                })
        except Exception:
            edges = []

    kegg_df = _safe_read_csv(output_dir / "kegg_pathways.csv")
    kegg = []
    if not kegg_df.empty:
        kegg = kegg_df[kegg_df.get("gene_id") == gene_id].to_dict(orient="records")

    eggnog_go_df = _safe_read_csv(output_dir / "eggnog_go_terms.csv")
    eggnog_go = []
    if not eggnog_go_df.empty:
        eggnog_go = eggnog_go_df[eggnog_go_df.get("gene_id") == gene_id].to_dict(orient="records")

    eggnog_dom_df = _safe_read_csv(output_dir / "eggnog_domains.csv")
    eggnog_domains = []
    if not eggnog_dom_df.empty:
        eggnog_domains = eggnog_dom_df[eggnog_dom_df.get("gene_id") == gene_id].to_dict(orient="records")

    interpro_df = _safe_read_csv(output_dir / "interpro_domains.csv")
    interpro = []
    if not interpro_df.empty:
        interpro = interpro_df[interpro_df.get("gene_id") == gene_id].to_dict(orient="records")

    secondary_structure = row.get("secondary_structure", "") if isinstance(row, pd.Series) else ""
    psipred_states = row.get("psipred_states", secondary_structure) if isinstance(row, pd.Series) else ""
    psipred_conf = []
    sec_conf = row.get("psipred_confidence") if isinstance(row, pd.Series) else None
    if isinstance(sec_conf, str) and sec_conf:
        try:
            psipred_conf = json.loads(sec_conf)
        except Exception:
            psipred_conf = []

    tmhmm_segments = []
    tmhmm_topology = row.get("tmhmm_topology", "") if isinstance(row, pd.Series) else ""
    tmhmm_field = row.get("tmhmm_segments") if isinstance(row, pd.Series) else None
    if isinstance(tmhmm_field, str) and tmhmm_field:
        try:
            tmhmm_segments = json.loads(tmhmm_field)
        except Exception:
            tmhmm_segments = []
    tmhmm_png_path = gene_dir / f"tmhmm_{gene_id}.png"
    tmhmm_png_url = (
        f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/tmhmm_{gene_id}.png"
        if tmhmm_png_path.exists()
        else None
    )
    tmhmm_txt_path = gene_dir / f"tmhmm_{gene_id}.txt"
    tmhmm_txt_content = ""
    tmhmm_txt_url = None
    if tmhmm_txt_path.exists():
        try:
            tmhmm_txt_content = tmhmm_txt_path.read_text(errors="ignore")
        except Exception:
            tmhmm_txt_content = ""
        tmhmm_txt_url = f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/tmhmm_{gene_id}.txt"

    tmhmm_flag = False
    raw_tmhmm_flag = row.get("tmhmm_is_transporter") if isinstance(row, pd.Series) else False
    if isinstance(raw_tmhmm_flag, str):
        tmhmm_flag = raw_tmhmm_flag.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        tmhmm_flag = bool(raw_tmhmm_flag)

    gor4_state_path = gene_dir / f"{gene_id}_mpsa_state.gif"
    gor4_profile_path = gene_dir / f"{gene_id}_mpsa1.gif"
    gor4_text_path = gene_dir / f"{gene_id}_gor4_raw.txt"
    gor4_state_url = (
        f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/{gene_id}_mpsa_state.gif"
        if gor4_state_path.exists()
        else None
    )
    gor4_profile_url = (
        f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/{gene_id}_mpsa1.gif"
        if gor4_profile_path.exists()
        else None
    )
    gor4_text_url = (
        f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/{gene_id}_gor4_raw.txt"
        if gor4_text_path.exists()
        else None
    )

    orthologs = []
    orth_field = row.get("orthologous_groups") if isinstance(row, pd.Series) else ""
    if isinstance(orth_field, str) and orth_field:
        orthologs = [o for o in orth_field.split(";") if o]

    rapid_fold_path = gene_dir / f"fold_rapid_{gene_id}.pdb"
    rapid_fold_url = (
        f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/fold_rapid_{gene_id}.pdb"
        if rapid_fold_path.exists()
        else None
    )

    detail = {
        "gene_id": gene_id,
        "log2FC": _sf(log2fc),
        "p_value": _sf(row.get("p_value")),
        "adj_p_value": _sf(row.get("adj_p_value")),
        "regulation": regulation,
        "protein_name": row.get("protein_name", "Unknown"),
        "uniprot_id": row.get("uniprot_id"),
        "function": row.get("function", "Not available"),
        "go_bp": row.get("GO_biological_process", ""),
        "go_mf": row.get("GO_molecular_function", ""),
        "go_cc": row.get("GO_cellular_component", ""),
        "interpro_domains": row.get("InterPro_domains", ""),
        "pfam_domains": row.get("Pfam_domains", ""),
        "cath_domains": row.get("CATH_domains", ""),
        "alphafold_confidence": row.get("alphafold_confidence", "N/A"),
        "alphafold_version": row.get("alphafold_version", "Unknown"),
        "secondary_structure": secondary_structure,
        "psipred_states": psipred_states,
        "psipred_confidence": psipred_conf,
        "fasta": fasta,
        "plain_sequence": plain_sequence,
        "pdb": pdb_text,
        "string_svg_url": svg_url,
        "string_edges": edges,
        "kegg": kegg,
        "eggnog_go": eggnog_go,
        "eggnog_domains": eggnog_domains,
        "interpro": interpro,
        "orthologs": orthologs,
        "rapid_fold_model": row.get("rapid_fold_model", ""),
        "rapid_fold_pdb": rapid_fold_url,
        "tmhmm_segments": tmhmm_segments,
        "tmhmm_topology": tmhmm_topology,
        "tmhmm_plot": tmhmm_png_url,
        "tmhmm_is_transporter": tmhmm_flag,
        "tmhmm_text": tmhmm_txt_content,
        "tmhmm_text_url": tmhmm_txt_url,
        "gor4_state_img": gor4_state_url,
        "gor4_profile_img": gor4_profile_url,
        "gor4_text_url": gor4_text_url,
        "place_location": row.get("place_location", ""),
        "string_edges_path": f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/string_edges_{gene_id}.csv" if edges_path.exists() else None,
        "pdb_path": f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/structure_{gene_id}.pdb" if pdb_path.exists() else None,
        "fasta_path": f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/protein_{gene_id}.fasta" if fasta_path.exists() else None,
    }
    return _clean_json(detail)


@app.post("/runs/{run_id}/gene/{gene_id}/phobius", response_class=JSONResponse)
async def gene_phobius(run_id: str, gene_id: str, force: bool = False):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run not completed")

    output_dir = info["output_dir"]
    ann_df = _safe_read_csv(output_dir / "annotations.csv")
    if ann_df.empty or "gene_id" not in ann_df.columns:
        raise HTTPException(status_code=404, detail="No annotation data")

    row = ann_df.loc[ann_df["gene_id"] == gene_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Gene not found")
    row = row.iloc[0]

    log2fc = row.get("log2FC", 0)
    regulation = "up" if pd.notna(log2fc) and log2fc > 0 else "down"
    gene_dir = output_dir / "genes" / ("upregulated" if regulation == "up" else "downregulated") / gene_id

    fasta_path = gene_dir / f"protein_{gene_id}.fasta"
    if not fasta_path.exists():
        raise HTTPException(status_code=404, detail="FASTA not found for gene")
    try:
        fasta_text = fasta_path.read_text()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read FASTA")

    seq = _clean_sequence(fasta_text)
    if not seq or len(seq) < 20:
        raise HTTPException(status_code=400, detail="Sequence too short or invalid for Phobius")

    try:
        result = _run_phobius(seq, gene_dir, gene_id, force=force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Phobius call failed: {exc}")

    txt_path = result.get("text_path")
    png_path = result.get("png_path")
    txt_url = None
    png_url = None
    if txt_path and txt_path.exists():
        txt_url = f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/{txt_path.name}"
    if png_path and png_path.exists():
        png_url = f"/runs/{run_id}/files/genes/{'upregulated' if regulation=='up' else 'downregulated'}/{gene_id}/{png_path.name}"

    return {
        "status": "ok",
        "cached": result.get("cached", False),
        "job_id": result.get("job_id"),
        "text": result.get("text", ""),
        "text_url": txt_url,
        "png_url": png_url,
    }


@app.post("/runs/{run_id}/gene/{gene_id}/fold", response_class=JSONResponse)
async def gene_fold(run_id: str, gene_id: str):
    info = manager.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run not completed")

    gene_dir = info["output_dir"] / "genes"
    fasta_path = gene_dir / "upregulated" / gene_id / f"protein_{gene_id}.fasta"
    if not fasta_path.exists():
        fasta_path = gene_dir / "downregulated" / gene_id / f"protein_{gene_id}.fasta"
    if not fasta_path.exists():
        raise HTTPException(status_code=404, detail="FASTA not found for gene")

    try:
        lines = fasta_path.read_text().splitlines()
        seq = "".join([ln.strip() for ln in lines if ln and not ln.startswith(">")])
    except Exception:
        raise HTTPException(status_code=500, detail="Unable to read sequence")

    out_path = fasta_path.parent / f"fold_rapid_{gene_id}.pdb"
    if out_path.exists():
        return {"status": "ok", "pdb_url": f"/runs/{run_id}/files/genes/{'upregulated' if 'upregulated' in str(out_path) else 'downregulated'}/{gene_id}/fold_rapid_{gene_id}.pdb"}

    pdb_file = _run_esm_fold(seq, out_path)
    if not pdb_file:
        raise HTTPException(status_code=502, detail="ESMFold (ESM Atlas) prediction failed or timed out")

    return {"status": "ok", "pdb_url": f"/runs/{run_id}/files/genes/{'upregulated' if 'upregulated' in str(out_path) else 'downregulated'}/{gene_id}/fold_rapid_{gene_id}.pdb"}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)