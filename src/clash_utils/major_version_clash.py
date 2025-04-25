#!/usr/bin/env python3
"""
Find node packs whose strict pins disagree on *major* versions.

Inputs
------
strict_nodes.csv        – columns: node_id, strict_deps, ...

Optional
--------
nodes_inventory.json    – to add a 'downloads' column (nice for ranking)

Outputs
-------
major_version_clashes.csv            (package-level summary)
strict_packs_with_major_clash.csv    (node-pack subset)
major_clash_pairs.csv               (node-by-node matrix of major-version clashes)

Example run
-----------
uv run major_version_clash.py --strict-csv strict_nodes.csv --inventory-json ../output/nodes_snapshot_20250420_202134.json
"""

from __future__ import annotations
import csv, json, sys, argparse
from pathlib import Path
from collections import defaultdict

from packaging.requirements import Requirement
from packaging.version import Version

def parse_arguments():
    parser = argparse.ArgumentParser(description="Find node packs with major version clashes in strict pins")
    parser.add_argument("--strict-csv", type=Path, default="strict_nodes.csv",
                        help="CSV file containing strict node information")
    parser.add_argument("--inventory-json", type=Path, default="nodes_inventory.json",
                        help="JSON file containing node inventory information (optional)")
    parser.add_argument("--output-pkg", type=Path, default="major_version_clashes.csv",
                        help="Output CSV file for package-level summary")
    parser.add_argument("--output-node", type=Path, default="strict_packs_with_major_clash.csv",
                        help="Output CSV file for node-pack subset")
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default="major_clash_pairs.csv",
        help="Node-by-node matrix of major-version clashes",
    )
    return parser.parse_args()

def parse_eq_pin(req_str: str):
    """Return (package, major) if req_str is 'package==x.y[.z]'; else None."""
    try:
        req = Requirement(req_str.strip())
    except Exception:
        return None
    if not req.specifier or next(iter(req.specifier)).operator != "==":
        return None
    ver = Version(next(iter(req.specifier)).version)
    return req.name.lower(), ver.major

def main(
    strict_csv: Path | str = "strict_nodes.csv",
    inventory_json: Path | str | None = "nodes_inventory.json",
    output_pkg: Path | str = "major_version_clashes.csv",
    output_node: Path | str = "strict_packs_with_major_clash.csv",
    output_matrix: Path | str = "major_clash_pairs.csv",
):
    """
    Finds major version clashes in strict dependencies and writes results to CSV files.

    Args:
        strict_csv: Path to the input CSV file with strict node dependencies.
        inventory_json: Optional path to the JSON inventory file for node metadata.
        output_pkg: Path for the output CSV summarizing package clashes.
        output_node: Path for the output CSV listing nodes involved in clashes.
        output_matrix: Path for the output CSV showing node-by-node clash details.
    """
    # Ensure inputs are Path objects
    STRICT_CSV = Path(strict_csv)
    INV_JSON = Path(inventory_json) if inventory_json else None
    OUT_PKG_CSV = Path(output_pkg)
    OUT_NODE_CSV = Path(output_node)
    OUT_MATRIX_CSV = Path(output_matrix)


    # ── 1 collect pins ────────────────────────────────────────────────────
    pkg_to_major_to_nodes = defaultdict(lambda: defaultdict(set))
    node_pkg_to_spec      = defaultdict(dict)          #  node → {pkg: '==x.y.z'}

    if not STRICT_CSV.exists():
        print(f"Error: Strict CSV file not found at {STRICT_CSV}", file=sys.stderr)
        return # Or raise an exception

    with STRICT_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            node = row["node_id"]
            for raw in row["strict_deps"].split(";"):
                parsed = parse_eq_pin(raw)
                if not parsed:
                    continue
                pkg, major = parsed
                spec_str   = raw.strip()
                pkg_to_major_to_nodes[pkg][major].add(node)
                node_pkg_to_spec[node][pkg] = spec_str

    # ── 2 packages with ≥2 majors ────────────────────────────────────────
    pkg_rows, clash_nodes = [], set()
    for pkg, majors in pkg_to_major_to_nodes.items():
        if len(majors) < 2:
            continue
        counts = {m: len(nodes) for m, nodes in majors.items()}
        pkg_rows.append({
            "package": pkg,
            "majors_pinned": "; ".join(f"{m}:{c}" for m,c in counts.items()),
            "packs_involved": sum(counts.values())
        })
        for nodes in majors.values():
            clash_nodes.update(nodes)

    # ── 3 write package-level summary ────────────────────────────────────
    if pkg_rows:
        pkg_rows.sort(key=lambda r: r["packs_involved"], reverse=True)
        OUT_PKG_CSV.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with OUT_PKG_CSV.open("w", newline="") as fh:
            fieldnames = list(pkg_rows[0].keys()) # Get fieldnames dynamically
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pkg_rows)

    # ── 4 write node-pack subset (optionally add downloads) ──────────────
    node_rows = [{"node_id": n} for n in sorted(clash_nodes)]
    if INV_JSON and INV_JSON.exists():
        try:
            inv = {item["id"]: item.get("downloads", 0)
                   for item in json.load(INV_JSON.open())}
            for r in node_rows:
                r["downloads"] = inv.get(r["node_id"], 0)
        except (FileNotFoundError, json.JSONDecodeError) as e:
             print(f"Warning: Could not load or parse inventory JSON {INV_JSON}: {e}", file=sys.stderr)


    if node_rows:
        OUT_NODE_CSV.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with OUT_NODE_CSV.open("w", newline="") as fh:
            fieldnames = list(node_rows[0].keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(node_rows)


    # --- 5 Write node-by-node clash matrix ---
    matrix_rows = []
    for pkg, majors in pkg_to_major_to_nodes.items():
        if len(majors) < 2:
            continue
        # build reverse index: node -> conflicting nodes for this pkg
        major_groups = list(majors.values())
        for group_idx, group in enumerate(major_groups):
            for node in group:
                conflicts = []
                spec_a = node_pkg_to_spec[node][pkg]
                # Compare with nodes in *other* major version groups for this package
                for other_group_idx, other_group in enumerate(major_groups):
                    if other_group_idx == group_idx: # Skip comparing group with itself
                        continue
                    for other_node in other_group:
                        spec_b = node_pkg_to_spec[other_node][pkg]
                        conflicts.append(f"{other_node}:{spec_b}")

                if conflicts: # Only add row if there are actual conflicts found
                    matrix_rows.append(
                        {
                            "node_id": node,
                            "package_spec": f"{pkg} {spec_a}",
                            "conflicting_nodes": "; ".join(conflicts),
                        }
                    )

    if matrix_rows:
        OUT_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        # Sort for consistent output
        matrix_rows.sort(key=lambda x: (x["node_id"], x["package_spec"]))
        with OUT_MATRIX_CSV.open("w", newline="") as fh:
            fieldnames = ["node_id", "package_spec", "conflicting_nodes"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matrix_rows)

    # --- 6 Print Summary ---
    print(f"✓ {len(pkg_rows)} libraries have ≥2 pinned major versions across {len(node_rows)} node packs.")
    print("  ↳ Detailed CSVs written:")
    if pkg_rows: print(f"    • {OUT_PKG_CSV}")
    if node_rows: print(f"    • {OUT_NODE_CSV}")
    if matrix_rows: print(f"    • {OUT_MATRIX_CSV}")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        strict_csv=args.strict_csv,
        inventory_json=args.inventory_json,
        output_pkg=args.output_pkg,
        output_node=args.output_node,
        output_matrix=args.output_matrix,
    )
