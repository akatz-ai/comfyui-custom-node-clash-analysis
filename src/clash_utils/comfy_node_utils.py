"""comfy_node_utils.py
====================
Utility helpers for working with the **ComfyUI custom‑node registry**.
Designed to be *imported* from a Jupyter notebook / Python REPL, but can also
be executed as a tiny CLI for quick one‑offs.

Typical notebook usage
----------------------
```python
from comfy_node_utils import snapshot_registry, build_inventory

# 1) Grab the latest registry snapshot (≈ 3–4 s)
snap_path = snapshot_registry()          # → data/nodes_snapshot_20250424_104501.json

# 2) Transform it into an "inventory" enriched with GitHub requirements.txt
inv_path  = build_inventory(snap_path)   # → data/nodes_inventory.json
```
Both helper functions return the *Path* to the JSON they wrote, so you can load
and explore the data straight away if you like:
```python
import json, pandas as pd
nodes = json.loads(inv_path.read_text())
(pd.json_normalize(nodes)
   .explode(["dependencies", "req_dependencies"])
   .head())
```

Key design points
-----------------
* **No CSV artefacts**
* Uses an env‑var `GITHUB_TOKEN` *if* present (to avoid GitHub API rate‑limits)
* Keeps network politeness – 100 req/min hard‑capped.
* Pure standard library + `requests` + `tqdm` (the latter is optional).
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv

import requests
from tqdm.auto import tqdm  # nicer bars inside notebooks

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

API_ROOT = "https://api.comfy.org/nodes"
PAGE_SIZE = 100                 # registry maximum as of 2025‑04
REQUEST_TIMEOUT = 20            # seconds
PAUSE_BETWEEN_CALLS = 0.15      # polite pause to avoid hammering the API
DATA_DIR = Path("data")         # default output folder (relative to CWD)
DATA_DIR.mkdir(exist_ok=True)

# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_page(page: int) -> Dict[str, Any]:
    """Return one page (latest versions only) from the registry."""
    params = {"page": page, "limit": PAGE_SIZE, "latest": True}
    r = requests.get(API_ROOT, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _sanitise_deps(raw_deps: List[str] | None) -> List[str]:
    """Strip comments / blank lines from dependency strings."""
    if not raw_deps:
        return []
    return [ln.strip() for ln in raw_deps if ln.strip() and not ln.strip().startswith("#")]


def _fetch_requirements(repo_url: str, *, token: Optional[str] = None) -> List[str]:
    """Fetch *requirements.txt* from a GitHub repo (if present)."""
    if not repo_url or "github.com" not in repo_url:
        return []

    # Extract <owner>/<repo>
    path_parts = urlparse(repo_url).path.lstrip("/").rstrip(".git").split("/")
    if len(path_parts) < 2:
        return []
    owner, repo = path_parts[:2]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/requirements.txt"
    headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        content = base64.b64decode(data["content"]).decode() if data.get("encoding") == "base64" else data["content"]
        return _sanitise_deps(content.splitlines())
    except Exception:
        return []  # swallow & continue – network blips shouldn't kill the run


# ── Public API ───────────────────────────────────────────────────────────────

def snapshot_registry(out_dir: Path = DATA_DIR, *, stamp: str | None = None,
                      node_limit: Optional[int] = None, show_stats: bool = True) -> Path:
    """Download the full registry (latest versions) and save a raw snapshot.

    Args
    ----
    out_dir        : Directory to write the snapshot file into. Created if needed.
    stamp          : Override the timestamp part of the filename. ISO‑like `YYYYMMDD_HHMMSS` is expected.
    node_limit     : If given, stop after *node_limit* nodes (handy for prototyping).
    show_stats     : Print a quick top‑deps table at the end.

    Returns
    -------
    Path to the newly written JSON file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot_path = out_dir / f"nodes_snapshot_{stamp}.json"

    print("📥  Downloading registry snapshot …")

    first_page = _fetch_page(1)
    total_pages = first_page["totalPages"]

    all_nodes: List[Dict[str, Any]] = []
    dep_counter: Counter[str] = Counter()

    processed = 0
    progress = tqdm(range(1, total_pages + 1), desc="Registry pages", unit="page")
    for page in progress:
        data = first_page if page == 1 else _fetch_page(page)
        for node in data["nodes"]:
            all_nodes.append(node)

            latest = node.get("latest_version") or {}
            dep_counter.update(_sanitise_deps(latest.get("dependencies")))

            processed += 1
            if node_limit and processed >= node_limit:
                progress.close()
                print(f"⚠️  Stopped early after {node_limit} nodes (node_limit reached).")
                break
        if node_limit and processed >= node_limit:
            break
        if page < total_pages:
            time.sleep(PAUSE_BETWEEN_CALLS)

    snapshot_path.write_text(json.dumps(all_nodes, indent=2), encoding="utf-8")
    print(f"✓ Snapshot written → {snapshot_path.relative_to(Path.cwd())}")

    if show_stats and dep_counter:
        print("\nTop 10 API‑declared dependency strings:")
        for pkg, cnt in dep_counter.most_common(10):
            print(f"  {pkg:<25} {cnt}")

    return snapshot_path


def build_inventory(snapshot_path: Path, *, out_path: Path | None = None,
                    github_token_env: str = "GITHUB_TOKEN", node_limit: Optional[int] = None,
                    show_stats: bool = True) -> Path:
    """Transform a *snapshot* into a slimmer `nodes_inventory.json`.

    Adds:
      • `downloads` (default 0 if absent)
      • `dependencies`  – from the registry's `latest_version`
      • `req_dependencies` – parsed from GitHub requirements.txt

    Args
    ----
    snapshot_path     : Path to a file produced by `snapshot_registry()`.
    out_path          : Custom output path; defaults to *data/nodes_inventory.json*.
    github_token_env  : Name of env‑var holding a GitHub personal‑access token.
    node_limit        : For quick tests – stop after N nodes.
    show_stats        : Print dependency frequency tables to stdout.

    Returns
    -------
    Path to the written inventory JSON.
    """
    if out_path is None:
        out_path = DATA_DIR / "nodes_inventory.json"

    raw_nodes: List[Dict[str, Any]] = json.loads(Path(snapshot_path).read_text())
    if node_limit:
        raw_nodes = raw_nodes[:node_limit]

    token = os.getenv(github_token_env)

    inventory: List[Dict[str, Any]] = []
    api_counter: Counter[str] = Counter()
    req_counter: Counter[str] = Counter()

    for node in tqdm(raw_nodes, desc="Processing nodes", unit="node"):
        latest = node.get("latest_version") or {}
        deps = _sanitise_deps(latest.get("dependencies"))
        repo_url = node.get("repository") or ""
        reqs = _fetch_requirements(repo_url, token=token)

        inventory.append({
            "id": node["id"],
            "name": node.get("name") or node["id"],
            "version": latest.get("version"),
            "repository": repo_url,
            "downloads": node.get("downloads", 0),
            "dependencies": deps,
            "req_dependencies": reqs,
        })

        api_counter.update(deps)
        req_counter.update(reqs)

    out_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"✓ Inventory written → {out_path.relative_to(Path.cwd())}")

    if show_stats:
        if api_counter:
            print("\nTop 10 API‑declared dependencies:")
            for pkg, cnt in api_counter.most_common(10):
                print(f"  {pkg:<25} {cnt}")
        if req_counter:
            print("\nTop 10 requirements.txt dependencies:")
            for pkg, cnt in req_counter.most_common(10):
                print(f"  {pkg:<25} {cnt}")

    return out_path


# ── Minimal CLI for convenience ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ComfyUI registry utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s_snap = sub.add_parser("snapshot", help="Create a registry snapshot")
    s_snap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    s_snap.add_argument("--node-limit", type=int)

    s_inv = sub.add_parser("inventory", help="Build inventory from a snapshot")
    s_inv.add_argument("snapshot", type=Path, help="Path to nodes_snapshot_…json")
    s_inv.add_argument("--out", type=Path)
    s_inv.add_argument("--node-limit", type=int)

    args = parser.parse_args()

    if args.cmd == "snapshot":
        snapshot_registry(out_dir=args.out_dir, node_limit=args.node_limit)
    elif args.cmd == "inventory":
        build_inventory(args.snapshot, out_path=args.out, node_limit=args.node_limit)
