"""
file_validation.py
===================
"""

from __future__ import annotations

import io
import os
import re
import socket
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------
# Configuration (overridable via environment variables, no code changes
# needed to tune limits per deployment)
# --------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}

MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)  # 25 MB
MAX_ZIP_ENTRIES = _env_int("MAX_ZIP_ENTRIES", 5_000)
MAX_ZIP_UNCOMPRESSED_BYTES = _env_int("MAX_ZIP_UNCOMPRESSED_BYTES", 200 * 1024 * 1024)
MAX_ZIP_COMPRESSION_RATIO = _env_int("MAX_ZIP_COMPRESSION_RATIO", 100)
MAX_ROWS = _env_int("MAX_UPLOAD_ROWS", 300_000)
MAX_COLUMNS = _env_int("MAX_UPLOAD_COLUMNS", 500)
SNIFF_BYTES = 65536  # bytes inspected for text/binary heuristics


class FileValidationError(ValueError):
    """Raised whenever an uploaded file fails a security/validity check."""


@dataclass
class ValidationResult:
    ok: bool
    reason: Optional[str] = None
    warnings: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Byte-level "magic number" signatures
# --------------------------------------------------------------------------
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"

# Signatures of file types that must NEVER be accepted, no matter what
# extension the upload claims to have.
_DISALLOWED_SIGNATURES = {
    b"MZ": "Windows executable (PE/EXE)",
    b"\x7fELF": "Linux/Unix executable (ELF)",
    b"\xca\xfe\xba\xbe": "Mach-O / Java class executable",
    b"\xfe\xed\xfa": "Mach-O executable",
    b"%PDF": "PDF document",
    b"\x1f\x8b": "gzip archive",
    b"7z\xbc\xaf\x27\x1c": "7-Zip archive",
    b"Rar!\x1a\x07": "RAR archive",
    b"#!": "shell/interpreter script (shebang)",
    b"\x89PNG": "PNG image",
    b"\xff\xd8\xff": "JPEG image",
    b"GIF8": "GIF image",
    b"BM": "Bitmap image",
}

_SUSPICIOUS_TEXT_PATTERNS = [
    re.compile(rb"<\s*script", re.IGNORECASE),
    re.compile(rb"<\?php", re.IGNORECASE),
    re.compile(rb"<%@", re.IGNORECASE),
    re.compile(rb"javascript\s*:", re.IGNORECASE),
    re.compile(rb"vbscript\s*:", re.IGNORECASE),
    re.compile(rb"on(error|load|click|mouseover)\s*=", re.IGNORECASE),
    re.compile(rb"powershell(\.exe)?\s", re.IGNORECASE),
    re.compile(rb"cmd(\.exe)?\s*/c", re.IGNORECASE),
    re.compile(rb"\bbase64_decode\b", re.IGNORECASE),
    re.compile(rb"\beval\s*\(", re.IGNORECASE),
    re.compile(rb"\bexec\s*\(", re.IGNORECASE),
    re.compile(rb"\bMSXML2\.XMLHTTP\b", re.IGNORECASE),
    re.compile(rb"\bWScript\.Shell\b", re.IGNORECASE),
]

# Formula-injection leading characters recognised by Excel/LibreOffice/
# Google Sheets when a "cell" is auto-executed as a formula. '+' and '-'
# are only suspicious when what follows isn't just a (very ordinary,
# very common in expression data) negative/positive number.
_FORMULA_LEAD_CHARS_STRICT = ("=", "@", "\t", "\r")
_FORMULA_LEAD_CHARS_NUMERIC_SIGN = ("+", "-")
_NUMERIC_FIELD_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?%?$")


def _is_formula_injection_field(field_value: str) -> bool:
    if not field_value:
        return False
    lead = field_value[0]
    if lead in _FORMULA_LEAD_CHARS_STRICT:
        return True
    if lead in _FORMULA_LEAD_CHARS_NUMERIC_SIGN:
        # Only flag if it's NOT an ordinary signed number, e.g. "-2.1"
        # (a completely normal log2FC value) is fine, but "-cmd|..." or
        # "+HYPERLINK(...)" is not.
        return not bool(_NUMERIC_FIELD_RE.match(field_value))
    return False

_MACRO_MARKERS = [
    b"vbaProject",
    b"vbaProject.bin",
    b"_VBA_PROJECT",
    b"Macros/VBA",
    "vba".encode("utf-16-le"),
    "macro".encode("utf-16-le"),
    b"macroEnabled",
]


# --------------------------------------------------------------------------
# Extension / filename helpers
# --------------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """Strip path separators/control chars; used only for display, never
    for building filesystem paths (the server always writes uploads under
    a server-generated name)."""
    name = (name or "upload").strip()
    name = name.replace("\x00", "")
    name = re.sub(r"[\\/]+", "_", name)
    name = re.sub(r"[\r\n\t]+", " ", name)
    name = name.strip(" .")
    return name[:200] or "upload"


def sanitize_form_text(value: Optional[str], max_len: int = 128) -> str:
    """Trim, strip control characters, and cap the length of small form
    fields (job name, column names, etc.) so they can't be used to smuggle
    oversized or control-character payloads into logs/metadata."""
    if not value:
        return ""
    value = value.strip()
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return value[:max_len]


def guess_extension(original_filename: str) -> str:
    ext = Path(sanitize_filename(original_filename)).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else ""


# --------------------------------------------------------------------------
# Core signature sniffing
# --------------------------------------------------------------------------
def _sniff_kind(head: bytes) -> str:
    for sig in _ZIP_SIGNATURES:
        if head.startswith(sig):
            return "zip"
    if head.startswith(_OLE_SIGNATURE):
        return "ole"
    for sig, label in _DISALLOWED_SIGNATURES.items():
        if head.startswith(sig):
            return f"blocked:{label}"
    return "text-candidate"


def _looks_like_text(sample: bytes) -> bool:
    if b"\x00" in sample:
        return False
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return False
    # Reject samples with a high proportion of non-printable bytes.
    control = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return control / max(len(sample), 1) < 0.01


def _scan_suspicious_text(sample: bytes) -> Optional[str]:
    for pattern in _SUSPICIOUS_TEXT_PATTERNS:
        if pattern.search(sample):
            return pattern.pattern.decode(errors="ignore")
    return None


def _contains_macro_markers(raw: bytes) -> bool:
    return any(marker in raw for marker in _MACRO_MARKERS)


# --------------------------------------------------------------------------
# Streamed save with a hard size cap (never trust Content-Length alone)
# --------------------------------------------------------------------------
async def stream_upload_to_disk(upload_file, dest_path: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Copy a FastAPI/Starlette UploadFile -> dest_path in chunks using the
    async read API (so the event loop is never blocked on disk/network
    I/O), aborting the moment max_bytes is exceeded so a lying or missing
    Content-Length header can't be used to exhaust disk/memory. Returns
    the number of bytes written."""
    total = 0
    chunk_size = 1024 * 1024
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise FileValidationError(
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
                    )
                out.write(chunk)
    except FileValidationError:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()
    return total


# --------------------------------------------------------------------------
# Per-type structural validation
# --------------------------------------------------------------------------
def _validate_zip_workbook(path: Path) -> None:
    """.xlsx is a zip archive - guard against zip bombs, macros, external
    links, and structurally invalid archives."""
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise FileValidationError(f"Not a valid Excel (.xlsx) archive: {exc}")

    with zf:
        infos = zf.infolist()
        if len(infos) == 0:
            raise FileValidationError("Excel file is empty or corrupted.")
        if len(infos) > MAX_ZIP_ENTRIES:
            raise FileValidationError("Excel file contains an unreasonable number of internal parts.")

        total_uncompressed = 0
        names = set()
        for info in infos:
            name = info.filename
            names.add(name)

            # Zip-slip guard (defensive; we never extract to disk).
            if name.startswith("/") or ".." in Path(name).parts:
                raise FileValidationError("Excel file contains unsafe internal paths.")

            total_uncompressed += info.file_size
            if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise FileValidationError("Excel file expands far beyond an expected size (possible zip bomb).")

            compressed = max(info.compress_size, 1)
            if info.file_size / compressed > MAX_ZIP_COMPRESSION_RATIO and info.file_size > 1_000_000:
                raise FileValidationError("Excel file contains an implausibly compressed part (possible zip bomb).")

            lowered = name.lower()
            if "vbaproject" in lowered or lowered.startswith("xl/macros"):
                raise FileValidationError("Macro-enabled Excel files are not allowed. Please save as a plain .xlsx.")
            if lowered.startswith("xl/externallinks/"):
                raise FileValidationError("Excel files with external workbook links are not allowed.")
            if lowered.startswith("xl/embeddings/") or lowered.startswith("xl/oleobjects/"):
                raise FileValidationError("Excel files with embedded objects are not allowed.")

        if "[Content_Types].xml" not in names:
            raise FileValidationError("Not a valid Excel (.xlsx) workbook (missing manifest).")
        if not any(n.startswith("xl/worksheets/") for n in names):
            raise FileValidationError("Not a valid Excel (.xlsx) workbook (no worksheets found).")

        try:
            content_types = zf.read("[Content_Types].xml").decode("utf-8", errors="ignore")
        except Exception:
            content_types = ""
        if "macroEnabled" in content_types or "vbaProject" in content_types:
            raise FileValidationError("Macro-enabled Excel files are not allowed.")


def _validate_legacy_xls(raw_head: bytes, full_bytes: bytes) -> None:
    """Legacy binary .xls (OLE Compound File). We don't fully parse the
    OLE directory (no extra dependency), so we fall back to conservative
    byte-pattern screening for known VBA/macro storage markers."""
    if not raw_head.startswith(_OLE_SIGNATURE):
        raise FileValidationError("Not a valid legacy Excel (.xls) file.")
    if _contains_macro_markers(full_bytes):
        raise FileValidationError("Macro-enabled legacy Excel files are not allowed. Please save as .xlsx or .csv.")


def _iter_xlsx_formula_cells(path: Path):
    """Yield (sheet, cell_coordinate, value) for any cell that openpyxl
    reports as a formula, without evaluating it."""
    import openpyxl  # already a hard dependency of this project

    wb = openpyxl.load_workbook(path, data_only=False, read_only=True)
    try:
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if cell.data_type == "f" or (
                        isinstance(cell.value, str) and cell.value.startswith("=")
                    ):
                        yield ws.title, cell.coordinate, cell.value
    finally:
        wb.close()


def _check_xlsx_formula_injection(path: Path, max_hits: int = 1) -> None:
    hits = 0
    for _sheet, _coord, _value in _iter_xlsx_formula_cells(path):
        hits += 1
        if hits >= max_hits:
            raise FileValidationError(
                "Spreadsheet contains formulas. For security reasons only plain values are accepted "
                "-- please paste values (Excel: 'Paste Special > Values') and re-upload."
            )


def _check_csv_formula_and_script_injection(path: Path, encoding: str) -> None:
    suspicious_hits = 0
    with path.open("r", encoding=encoding, errors="replace", newline="") as fh:
        for line_no, line in enumerate(fh):
            if line_no > MAX_ROWS + 5:
                break
            stripped = line.lstrip(" \ufeff")
            for field_value in stripped.split(","):
                field_value = field_value.strip().strip('"')
                if len(field_value) > 1 and _is_formula_injection_field(field_value):
                    raise FileValidationError(
                        "File contains spreadsheet-formula-style values (starting with =, +, -, or @). "
                        "For security reasons only plain data values are accepted."
                    )
            if _scan_suspicious_text(line.encode("utf-8", errors="ignore")):
                suspicious_hits += 1
                if suspicious_hits > 0:
                    raise FileValidationError(
                        "File contains embedded script-like content and was rejected for security reasons."
                    )


def _detect_text_encoding(sample: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def validate_uploaded_file(path: Path, declared_ext: str) -> ValidationResult:
    """Run the full validation pipeline against an already-saved file.

    Raises FileValidationError on any failure; returns a ValidationResult
    (ok=True) with any non-fatal warnings otherwise.
    """
    warnings: list = []

    if declared_ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            "Unsupported file type. Please upload a .csv, .tsv, .txt, .xlsx, or .xls file."
        )

    size = path.stat().st_size
    if size == 0:
        raise FileValidationError("The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise FileValidationError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")

    with path.open("rb") as fh:
        head = fh.read(SNIFF_BYTES)

    kind = _sniff_kind(head)
    if kind.startswith("blocked:"):
        raise FileValidationError(
            f"Upload rejected: file content matches a {kind.split(':', 1)[1]}, not a data table."
        )

    if declared_ext in {".xlsx"}:
        if kind != "zip":
            raise FileValidationError("This does not look like a valid .xlsx file (wrong file signature).")
        _validate_zip_workbook(path)
        _check_xlsx_formula_injection(path)

    elif declared_ext == ".xls":
        if kind != "ole":
            raise FileValidationError("This does not look like a valid legacy .xls file (wrong file signature).")
        full_bytes = path.read_bytes()
        _validate_legacy_xls(head, full_bytes)
        warnings.append(
            "Legacy .xls files receive limited macro screening compared to .xlsx/.csv. "
            "If possible, re-save as .xlsx or .csv."
        )

    else:  # .csv / .tsv / .txt
        if kind in ("zip", "ole"):
            raise FileValidationError(
                "This does not look like a plain-text CSV/TSV/TXT file "
                "(it appears to be a binary/Excel container)."
            )
        if not _looks_like_text(head):
            raise FileValidationError("File does not appear to be valid plain-text tabular data.")
        encoding = _detect_text_encoding(head)
        _check_csv_formula_and_script_injection(path, encoding)

    return ValidationResult(ok=True, warnings=warnings)


def enforce_dataframe_limits(row_count: int, column_count: int) -> None:
    if row_count > MAX_ROWS:
        raise FileValidationError(
            f"File has {row_count:,} rows, which exceeds the {MAX_ROWS:,} row limit for a single run."
        )
    if column_count > MAX_COLUMNS:
        raise FileValidationError(
            f"File has {column_count:,} columns, which exceeds the {MAX_COLUMNS:,} column limit."
        )


# --------------------------------------------------------------------------
# Optional ClamAV integration - fully inert unless a clamd daemon is
# actually reachable on the host. No package needs to be pip-installed for
# this to stay dormant; if you later run ClamAV's `clamd` service, this
# will automatically start scanning every upload with it.
# --------------------------------------------------------------------------
_CLAMD_HOST = os.getenv("CLAMD_HOST", "127.0.0.1")
_CLAMD_PORT = _env_int("CLAMD_PORT", 3310)
_CLAMD_TIMEOUT = 5


def clamav_available() -> bool:
    try:
        with socket.create_connection((_CLAMD_HOST, _CLAMD_PORT), timeout=1) as sock:
            sock.sendall(b"zPING\0")
            resp = sock.recv(64)
            return resp.startswith(b"PONG")
    except OSError:
        return False


def scan_with_clamav(path: Path) -> Optional[str]:
    """Return None if clean/unavailable, or a threat description string if
    clamd reports the file as infected. Speaks clamd's lightweight
    "INSTREAM" protocol directly over a socket - no client library needed."""
    if not clamav_available():
        return None
    try:
        with socket.create_connection((_CLAMD_HOST, _CLAMD_PORT), timeout=_CLAMD_TIMEOUT) as sock:
            sock.sendall(b"zINSTREAM\0")
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    sock.sendall(len(chunk).to_bytes(4, "big") + chunk)
            sock.sendall((0).to_bytes(4, "big"))
            response = b""
            while True:
                part = sock.recv(4096)
                if not part:
                    break
                response += part
                if b"\0" in part:
                    break
        text = response.decode("utf-8", errors="ignore").strip("\x00").strip()
        if "FOUND" in text and "OK" not in text.split(":")[-1]:
            return text
        return None
    except OSError:
        return None
