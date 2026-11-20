"""
security.py
============
Small, dependency-free security helpers shared by server.py:

- Trusted-proxy-aware client IP resolution (so a random internet client
  can't just spoof X-Forwarded-For to impersonate another user's IP,
  while still correctly reading the real visitor IP when the request
  really did come from your reverse proxy).
- A lightweight in-memory per-IP rate limiter for the /run submission
  endpoint.
- Security response headers middleware.
"""

from __future__ import annotations

import ipaddress
import os
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# --------------------------------------------------------------------------
# Trusted proxy configuration
# --------------------------------------------------------------------------
# Only requests whose *direct* TCP peer is one of these addresses are
# allowed to supply X-Forwarded-For / X-Real-IP. Everyone else's headers
# are ignored and we fall back to the direct socket peer address. This is
# what stops an attacker from just setting X-Forwarded-For themselves.
#
# Default covers "reverse proxy on the same host" (the common case for
# nginx/Caddy in front of uvicorn). Add your proxy's address via the
# TRUSTED_PROXIES env var (comma separated) if it runs elsewhere, e.g.
# a Docker bridge network or a load balancer with a fixed internal IP.
def _load_trusted_proxies() -> set:
    raw = os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1")
    proxies = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            proxies.add(str(ipaddress.ip_address(item)))
        except ValueError:
            # Allow CIDR ranges too, e.g. 10.0.0.0/8 for a docker network.
            try:
                proxies.add(ipaddress.ip_network(item, strict=False))
            except ValueError:
                continue
    return proxies


TRUSTED_PROXIES = _load_trusted_proxies()


def _is_trusted_proxy(addr: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(addr)
    except ValueError:
        return False
    for entry in TRUSTED_PROXIES:
        if isinstance(entry, str):
            if addr == entry:
                return True
        else:
            if ip_obj in entry:
                return True
    return False


def _valid_ip(value: str) -> str:
    value = (value or "").strip()
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return ""


def extract_client_ip(request: Request) -> str:
    """Resolve the real end-user IP address for this request.

    Only trusts X-Forwarded-For / X-Real-IP when the request arrived
    directly from a configured trusted proxy; otherwise the direct
    socket peer is used, which prevents header spoofing from the open
    internet while still supporting a standard reverse-proxy deployment.
    """
    direct_peer = request.client.host if request.client else ""

    if _is_trusted_proxy(direct_peer):
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            # Standard convention: proxies APPEND to this header, so the
            # left-most entry is the original client.
            candidate = forwarded_for.split(",")[0].strip()
            valid = _valid_ip(candidate)
            if valid:
                return valid
        real_ip = request.headers.get("x-real-ip", "")
        valid = _valid_ip(real_ip)
        if valid:
            return valid

    valid = _valid_ip(direct_peer)
    return valid or "unknown"


# --------------------------------------------------------------------------
# Simple in-memory sliding-window rate limiter
# --------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_events: int, window_seconds: float):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] > self.window_seconds:
                events.popleft()
            if len(events) >= self.max_events:
                return False
            events.append(now)
            return True

    def retry_after(self, key: str) -> float:
        with self._lock:
            events = self._events.get(key)
            if not events:
                return 0.0
            return max(0.0, self.window_seconds - (time.monotonic() - events[0]))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


RUN_SUBMIT_LIMITER = RateLimiter(
    max_events=_env_int("RUN_RATE_LIMIT_PER_HOUR", 12),
    window_seconds=3600,
)

CHAT_MESSAGE_LIMITER = RateLimiter(
    max_events=_env_int("CHAT_RATE_LIMIT_PER_HOUR", 60),
    window_seconds=3600,
)

CHAT_BURST_LIMITER = RateLimiter(
    max_events=_env_int("CHAT_RATE_LIMIT_PER_MINUTE", 8),
    window_seconds=60,
)


# --------------------------------------------------------------------------
# Security response headers
# --------------------------------------------------------------------------
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' "
    "https://cdn.plot.ly https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data: blob: https:; "
    "connect-src 'self' https://alphafold.ebi.ac.uk https://alphasync.stjude.org "
    "https://www.uniprot.org https://rest.ensembl.org https://string-db.org "
    "https://rest.kegg.jp https://www.ebi.ac.uk https://cdn.jsdelivr.net; "
    "frame-src https://www.youtube.com https://alphasync.stjude.org; "
    "form-action 'self' https://blast.ncbi.nlm.nih.gov; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'self';"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard hardening headers to every response."""

    def __init__(self, app, force_https: bool = False):
        super().__init__(app)
        self.force_https = force_https

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=()",
        )
        response.headers.setdefault("Content-Security-Policy", _CSP)
        if self.force_https:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        return response
