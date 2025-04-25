#!/usr/bin/env python3
"""
Analyse ComfyUI node-pack metadata for *strict* dependency pins.

Strict = any requirement that contains either:
    • '=='  (exact equality)
    • '<='  (maximum upper bound)
    • '<'   (exclusive upper bound)

Outputs:
  • strict_pins.csv  – package-level stats (counts & both conflict metrics)
  • strict_nodes.csv – node-level stats with strict dependency details
  • summary.txt      – ecosystem-level headline numbers with both metrics
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse

import pandas as pd
from packaging.requirements import Requirement, InvalidRequirement
from packaging.specifiers import SpecifierSet

# ── helpers ───────────────────────────────────────────────────────────
STRICT_OPS = {"==", "<=", "<"}

def clean_requirement_string(original_req_str: str) -> str:
    """
    Clean a requirement string by:
    - Stripping whitespace and BOM character
    - Removing inline comments (anything after #)
    Returns the cleaned string
    """
    # Strip whitespace and BOM character
    req_str = original_req_str.strip().lstrip('\ufeff')
    
    # Handle inline comments: remove '#' and anything after it
    if ' #' in req_str:
        req_str = req_str.split(' #', 1)[0].strip()
        
    return req_str

def validate_requirements(req_dependencies: List[str], node_id: str) -> bool:
    """
    Validates if all requirement strings in the list are parsable according
    to standard Python packaging rules (PEP 508) or are valid requirements.txt lines.

    Allows for comments, empty lines, and common directives like -r, -e, file paths,
    and VCS links. Handles inline comments.
    Returns True if all lines are valid, False otherwise.
    """
    for original_req_str in req_dependencies:
        # Clean the requirement string
        req_str = clean_requirement_string(original_req_str)

        # Skip empty lines and full-line comments
        if not req_str or req_str.startswith('#'):
            continue

        # Allow common directives, local paths, VCS links, and direct URLs
        if req_str.startswith(('-', '.', '/', 'git+', 'http:', 'https:')):
            continue

        # Attempt to parse as a standard PEP 508 requirement string
        try:
            # Check if the requirement string is now empty after comment removal
            if not req_str:
                continue
            Requirement(req_str)
        except InvalidRequirement:
            return False  # Found an invalid requirement format

    return True  # All lines were valid requirements, directives, comments, or empty

def is_strict(req_str: str) -> Tuple[bool, str, str]:
    """
    Return (is_strict, package_name, spec_repr)
    • spec_repr is a canonical version bucket such as '==1.2.3' or '<=1.0'
    """
    original_req_str = req_str  # Keep for error messages
    # Clean the requirement string
    req_str = clean_requirement_string(req_str)

    # Ignore empty lines or lines that are clearly not standard requirements
    if not req_str or req_str.startswith(('-', '.', '/', 'git+', 'http:', 'https:')):
        return False, "", ""

    try:
        req = Requirement(req_str)
    except Exception:
        # Skip unparsable requirements
        return False, "<unparsed>", original_req_str

    # no specifiers → not strict
    if not req.specifier:
        return False, req.name.lower(), ""

    # Check if *any* specifier uses a strict operator
    for spec in req.specifier:
        if spec.operator in STRICT_OPS:
            # Return the first strict specifier found
            return True, req.name.lower(), f"{spec.operator}{spec.version}"

    # No strict specifier found
    return False, req.name.lower(), ""

def specs_conflict(a: str, b: str) -> bool:
    """
    Precise compatibility check:
    • identical strings   → compatible
    • one side '=='       → test that exact version against the other side
    • both sides <= / <   → compatible (they overlap below the narrower cap)
    """
    if a == b:
        return False

    sa, sb = SpecifierSet(a), SpecifierSet(b)

    # fast path: if either side pins exactly, test membership
    if "==" in a:
        v = a.split("==")[1]
        return not sb.contains(v, prereleases=True)
    if "==" in b:
        v = b.split("==")[1]
        return not sa.contains(v, prereleases=True)

    # both are range caps (<= / <) → they always overlap under the stricter cap
    return False

def main(json_path: Path, top_conflicts: int, out_csv: Path = Path("strict_pins.csv"), 
         out_nodes_csv: Path = Path("strict_nodes.csv"), out_txt: Path = Path("summary.txt")) -> None:
    data = json.loads(json_path.read_text())
    pkg_buckets: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    node_with_strict: Set[str] = set()
    nodes_with_invalid_reqs: Set[str] = set()
    
    # Track detailed node data
    node_data = {}

    for node in data:
        node_id = node["id"]
        # Validate requirements
        is_valid = validate_requirements(node.get("req_dependencies", []), node_id)
        if not is_valid:
            nodes_with_invalid_reqs.add(node_id)
            continue
        
        # Initialize node data record
        strict_deps = []
            
        for raw in node.get("req_dependencies", []):
            is_s, pkg, bucket = is_strict(raw)
            if is_s:
                pkg_buckets[pkg][bucket].add(node_id)
                node_with_strict.add(node_id)
                strict_deps.append(f"{pkg}{bucket}")
        
        # Store data for nodes with strict dependencies
        if strict_deps:
            node_data[node_id] = {
                "node_id": node_id,
                "num_strict_deps": len(strict_deps),
                "strict_deps": "; ".join(strict_deps),
                "total_deps": len(node.get("req_dependencies", [])),
            }

    # ── per-package stats ────────────────────────────────────────────
    rows = []
    for pkg, buckets in pkg_buckets.items():
        totals = {k: len(v) for k, v in buckets.items()}
        total_nodes = sum(totals.values())
        
        # Naive conflict calculation (from strict_pin_audit.py)
        if len(totals) > 1:
            bucket_sizes = list(totals.values())
            total_pairs = total_nodes * (total_nodes - 1) // 2
            same_pairs = sum(n * (n - 1) // 2 for n in bucket_sizes)
            conflict_pairs = total_pairs - same_pairs
        else:
            conflict_pairs = 0
            
        # Precise conflict calculation (from strict_pin_audit_precise.py)
        conflict_precise = 0
        spec_items = list(buckets.items())
        for i in range(len(spec_items)):
            spec_i, nodes_i = spec_items[i]
            for j in range(i + 1, len(spec_items)):
                spec_j, nodes_j = spec_items[j]
                if specs_conflict(spec_i, spec_j):
                    conflict_precise += len(nodes_i) * len(nodes_j)

        rows.append(
            {
                "package": pkg,
                "node_packs_strict": total_nodes,
                "distinct_strict_specs": len(totals),
                "conflict_pairs": conflict_pairs,
                "precise_conflicts": conflict_precise,
                "per_spec_counts": "; ".join(f"{k}:{v}" for k, v in totals.items()),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["precise_conflicts", "conflict_pairs", "node_packs_strict"], ascending=False
    )
    df.to_csv(out_csv, index=False)

    # Calculate conflict data per node
    node_conflicts = defaultdict(set)
    
    # Find which nodes conflict with other nodes
    for pkg, buckets in pkg_buckets.items():
        spec_items = list(buckets.items())
        for i in range(len(spec_items)):
            spec_i, nodes_i = spec_items[i]
            for j in range(i + 1, len(spec_items)):
                spec_j, nodes_j = spec_items[j]
                if specs_conflict(spec_i, spec_j):
                    for node_i in nodes_i:
                        for node_j in nodes_j:
                            node_conflicts[node_i].add(node_j)
                            node_conflicts[node_j].add(node_i)
    
    # Update node data with conflict information
    for node_id in node_data:
        conflicts = node_conflicts.get(node_id, set())
        node_data[node_id]["conflict_count"] = len(conflicts)
        node_data[node_id]["conflict_percentage"] = len(conflicts) / len(node_with_strict) * 100 if len(node_with_strict) > 0 else 0
    
    # Create and save node-level CSV
    nodes_df = pd.DataFrame(list(node_data.values()))
    if not nodes_df.empty:
        nodes_df = nodes_df.sort_values(["conflict_count", "num_strict_deps"], ascending=False)
        nodes_df.to_csv(out_nodes_csv, index=False)
        print(f"✓ wrote {out_nodes_csv}")

    # ── summary ──────────────────────────────────────────────────────
    total_nodes = len(data)
    total_nodes_with_strict = len(node_with_strict)
    total_conflicts = df["conflict_pairs"].sum()
    total_precise_conflicts = df["precise_conflicts"].sum()
    total_possible_pairs = total_nodes_with_strict * (total_nodes_with_strict - 1) // 2
    
    conflict_percentage = (total_conflicts / total_possible_pairs * 100) if total_possible_pairs > 0 else 0
    precise_conflict_percentage = (total_precise_conflicts / total_possible_pairs * 100) if total_possible_pairs > 0 else 0
    
    with out_txt.open("w") as fh:
        fh.write(f"Total node packs analysed: {total_nodes}\n")
        fh.write(f"Node packs with ≥1 strict pin: {len(node_with_strict)} "
                 f"({len(node_with_strict)/total_nodes:.1%})\n")
        fh.write(f"Node packs with invalid requirements: {len(nodes_with_invalid_reqs)} "
                 f"({len(nodes_with_invalid_reqs)/total_nodes:.1%})\n\n")
        
        fh.write(f"Total conflicts across all packages (naive): {total_conflicts}\n")
        fh.write(f"Total precise conflicts across all packages: {total_precise_conflicts}\n")
        fh.write(f"Total possible pairs across all packages: {total_possible_pairs}\n")
        fh.write(f"Percentage of pairs in conflict (naive): {conflict_percentage:.2f}%\n")
        fh.write(f"Percentage of pairs in conflict (precise): {precise_conflict_percentage:.2f}%\n\n")
        
        fh.write(f"Top {top_conflicts} conflict-generating packages:\n")
        for _, r in df.head(top_conflicts).iterrows():
            fh.write(f"  {r.package:<20} "
                     f"packs={r.node_packs_strict:<4} "
                     f"specs={r.distinct_strict_specs:<2} "
                     f"naive_conflicts={r.conflict_pairs:<6} "
                     f"precise_conflicts={r.precise_conflicts}\n")

    print(f"✓ wrote {out_csv}")
    print(f"✓ wrote {out_txt}")
    print("\n" + out_txt.read_text())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse ComfyUI node-pack metadata for strict dependency pins with both naive and precise conflict metrics."
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=Path("nodes_inventory.json"),
        help="Path to the nodes_inventory.json file.",
    )
    parser.add_argument(
        "--top-conflicts",
        type=int,
        default=15,
        help="Number of top conflict-generating packages to display (default: 15)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("strict_pins.csv"),
        help="Path for the package-level stats CSV output (default: strict_pins.csv)",
    )
    parser.add_argument(
        "--out-nodes-csv",
        type=Path,
        default=Path("strict_nodes.csv"),
        help="Path for the node-level stats CSV output (default: strict_nodes.csv)",
    )
    parser.add_argument(
        "--out-txt",
        type=Path,
        default=Path("summary.txt"),
        help="Path for the summary text output (default: summary.txt)",
    )
    args = parser.parse_args()

    if not args.json_file.exists():
        sys.exit(f"❌ {args.json_file} not found – please provide a valid path using --json-file.")
    main(args.json_file, args.top_conflicts, args.out_csv, args.out_nodes_csv, args.out_txt)