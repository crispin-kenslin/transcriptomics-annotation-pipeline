from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

import pandas as pd

from services.rag import (
    build_dataset_context,
    build_fallback_answer,
    build_messages,
    stream_gemini,
)

MAX_QUESTION_LENGTH = 1500
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_MESSAGE_LENGTH = 4000
MAX_HISTORY_STORE = 100

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class ChatInputError(ValueError):
    pass


def sanitize_question(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ChatInputError("Question must be text.")
    text = _CONTROL_CHARS_RE.sub("", raw).strip()
    if not text:
        raise ChatInputError("Question cannot be empty.")
    if len(text) > MAX_QUESTION_LENGTH:
        raise ChatInputError(f"Question is too long (max {MAX_QUESTION_LENGTH} characters).")
    return text


def sanitize_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in raw[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        text = _CONTROL_CHARS_RE.sub("", content).strip()[:MAX_HISTORY_MESSAGE_LENGTH]
        if text:
            cleaned.append({"role": role, "content": text})
    return cleaned


def clamp_threshold(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return max(low, min(parsed, high))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _history_path(run_dir: Path) -> Path:
    return Path(run_dir) / "chat_history.json"


def load_chat_history(run_dir: Path) -> list[dict[str, Any]]:
    path = _history_path(run_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def append_chat_turn(run_dir: Path, question: str, answer: str, citations: list[dict[str, Any]]) -> None:
    history = load_chat_history(run_dir)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    history.append({"role": "user", "content": question, "timestamp": now})
    history.append({"role": "assistant", "content": answer, "timestamp": now, "citations": citations})
    if len(history) > MAX_HISTORY_STORE:
        history = history[-MAX_HISTORY_STORE:]
    try:
        _history_path(run_dir).write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def clear_chat_history(run_dir: Path) -> None:
    path = _history_path(run_dir)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


async def stream_chat_answer(
    output_dir: Path,
    run_dir: Path,
    species_params: dict[str, Any],
    question: str,
    conversation_history: list[dict[str, str]],
    log2fc_threshold: float,
    pval_threshold: float,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncGenerator[dict[str, Any], None]:
    ann_df = _safe_read_csv(Path(output_dir) / "annotations.csv")
    kegg_df = _safe_read_csv(Path(output_dir) / "kegg_pathways.csv")
    up_df = _safe_read_csv(Path(output_dir) / "upregulated_genes.csv")
    down_df = _safe_read_csv(Path(output_dir) / "downregulated_genes.csv")

    if ann_df.empty:
        yield {
            "event": "done",
            "answer": "This run does not have any annotated genes yet, so I don't have data to answer from.",
            "citations": [],
            "genes_annotated": [],
        }
        return

    context_text, _sources = build_dataset_context(ann_df, kegg_df, up_df, down_df, question, limit=8)
    messages = build_messages(question, context_text, conversation_history)

    answer_parts: list[str] = []
    model_options = {"temperature": 0.3, "top_p": 0.9, "max_output_tokens": 2048}

    async for token in stream_gemini(messages, model_options):
        if await is_disconnected():
            break
        answer_parts.append(token)
        yield {"event": "token", "token": token}

    answer = "".join(answer_parts).strip()
    if not answer:
        answer = build_fallback_answer(question, context_text, _sources)

    append_chat_turn(run_dir, question, answer, [])

    yield {
        "event": "done",
        "answer": answer,
        "citations": [],
        "genes_annotated": [],
    }
