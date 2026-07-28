"""Minimal local web UI for runs / traces / costs (FastAPI static page)."""

from __future__ import annotations

import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>aire UI</title>
  <style>
    :root { --bg:#0f1419; --fg:#e7ecf3; --muted:#8b98a8; --accent:#3d9cf0; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin:0;
           background:var(--bg); color:var(--fg); }
    header { padding:1.25rem 1.5rem; border-bottom:1px solid #243040;
             display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }
    h1 { margin:0; font-size:1.25rem; letter-spacing:0.02em; }
    label { color:var(--muted); font-size:0.8rem; }
    input, select { background:#161d27; color:var(--fg); border:1px solid #243040;
                    border-radius:6px; padding:0.35rem 0.5rem; }
    main { display:grid; grid-template-columns:1fr 1fr; gap:1rem; padding:1.5rem; }
    section { background:#161d27; border:1px solid #243040; border-radius:8px; padding:1rem; }
    h2 { margin:0 0 0.75rem; font-size:0.95rem; color:var(--accent); }
    pre { margin:0; white-space:pre-wrap; word-break:break-word; color:var(--muted);
          font-size:0.8rem; max-height:320px; overflow:auto; }
    @media (max-width:800px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>aire — runs / traces / costs</h1>
    <label>trace filter <input id="traceFilter" placeholder="span name"/></label>
    <label>limit <input id="traceLimit" type="number" value="50" min="1"
           style="width:4rem"/></label>
  </header>
  <main>
    <section><h2>Traces</h2><pre id="traces">loading…</pre></section>
    <section><h2>Costs</h2><pre id="costs">loading…</pre></section>
    <section><h2>Metrics</h2><pre id="metrics">loading…</pre></section>
    <section><h2>Events</h2><pre id="events">loading…</pre></section>
    <section><h2>Manifest</h2><pre id="manifest">loading…</pre></section>
  </main>
  <script>
    async function load(id, path) {
      try {
        const r = await fetch(path);
        document.getElementById(id).textContent = JSON.stringify(await r.json(), null, 2);
      } catch (e) {
        document.getElementById(id).textContent = String(e);
      }
    }
    function tracesUrl() {
      const name = document.getElementById('traceFilter').value.trim();
      const limit = document.getElementById('traceLimit').value || '50';
      const q = new URLSearchParams({limit});
      if (name) q.set('name', name);
      return '/api/traces?' + q.toString();
    }
    function refresh() {
      load('traces', tracesUrl());
      load('costs', '/api/costs');
      load('metrics', '/api/metrics');
      load('events', '/api/events');
    }
    load('manifest', '/api/manifest');
    refresh();
    document.getElementById('traceFilter').addEventListener('change', refresh);
    document.getElementById('traceLimit').addEventListener('change', refresh);
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


def create_ui_app(  # noqa: C901
    *,
    runtime: Any | None = None,
    title: str = "aire UI",
) -> Any:
    """FastAPI app serving a minimal ops dashboard. Requires ``aire[serve]``."""
    if importlib.util.find_spec("fastapi") is None:
        raise ConfigurationError(
            "fastapi required for aire UI: pip install 'aire[serve]'",
            code="ui.fastapi_missing",
        )
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    from aire._version import __version__
    from aire.ai import AI, default_runtime

    rt = runtime or default_runtime()
    app = FastAPI(title=title, version=__version__)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _HTML

    @app.get("/api/traces")
    async def traces(name: str | None = None, limit: int = 50) -> dict[str, Any]:
        return {"traces": AI.observe.traces(name=name, limit=limit)}

    @app.get("/api/costs")
    async def costs() -> dict[str, Any]:
        return {"costs": AI.observe.costs()}

    @app.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        m = AI.observe.metrics
        if hasattr(m, "snapshot"):
            snapshot = m.snapshot()
        elif hasattr(m, "describe"):
            snapshot = m.describe()
        else:
            snapshot = {}
        return {"metrics": snapshot}

    @app.get("/api/events")
    async def events(pattern: str | None = None) -> dict[str, Any]:
        return {"events": AI.observe.events(pattern)}

    @app.get("/api/manifest")
    async def manifest() -> dict[str, Any]:
        return {
            "library": "aire",
            "version": __version__,
            "runtime": rt.describe() if hasattr(rt, "describe") else {},
            "ai": AI.describe(),
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def describe() -> dict[str, Any]:
    return {
        "kind": "ui",
        "available": importlib.util.find_spec("fastapi") is not None,
        "install": "pip install 'aire[serve]'",
        "factory": "aire.ui.create_ui_app",
        "endpoints": ["/api/traces", "/api/costs", "/api/metrics", "/api/events", "/api/manifest"],
    }
