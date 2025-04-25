#!/usr/bin/env python3
"""
Probability a user bumps into a *major-version* clash for each package,
assuming they install **n** node-packs (default n=2), weighted by downloads.

Inputs
------
nodes_inventory.json         – registry inventory with download counts
major_clash_pairs.csv        – node_id, package_spec, conflicting_nodes

Outputs
-------
clash_probability_by_pkg.csv – package, p_pair, p_n, percent_n

Console table shows the top packages sorted by p_n.

Usage examples
--------------
# default: two random packs
$ python clash_probabilities.py --inventory-json path/to/nodes_inventory.json

# user installs ~10 packs, custom clash pairs, four-decimal precision
$ python clash_probabilities.py --inventory-json inv.json --clash-pairs-csv clashes.csv --n 10 --prec 4
"""

from __future__ import annotations
import csv, json, argparse, math, sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Set, List

# Make Pandas optional only if core logic could be rewritten without it,
# but it's heavily used here for sorting and output. Keep required for now.
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    # print("Error: pandas library is required for this script.", file=sys.stderr)
    # print("       Install it with: pip install pandas", file=sys.stderr)
    # sys.exit(1) # Or handle differently if parts can run without pandas


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def parse_conflict_list(raw: str) -> list[Tuple[str, str]]:
    """'A:torch==2.0 ; B:torch==1.13' -> [(A,'torch==2.0'), (B,'torch==1.13')]"""
    out: List[Tuple[str, str]] = []
    for piece in raw.split(";"):
        piece = piece.strip()
        if piece:
             # Handle potential missing colon gracefully
            if ":" in piece:
                node, spec = piece.split(":", 1)
                out.append((node.strip(), spec.strip()))
            else:
                # Log or ignore malformed piece
                # print(f"Warning: Malformed conflict piece '{piece}'", file=sys.stderr)
                pass
    return out

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate clash probability by package based on downloads.")
    parser.add_argument("--inventory-json", type=Path, default="nodes_inventory.json",
                        help="Input JSON node inventory file with download counts.")
    parser.add_argument("--clash-pairs-csv", type=Path, default="major_clash_pairs.csv",
                        help="Input CSV file with node clash pairs.")
    parser.add_argument("--output-csv", type=Path, default="clash_probability_by_pkg.csv",
                         help="Output CSV file path for probabilities.")
    parser.add_argument("--n", type=int, default=2,
                        help="Assumed number of packs a user installs (for p_n calculation).")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Number of top packages to list in the console output.")
    parser.add_argument("--prec", type=int, default=2,
                        help="Decimal places for percentages in console output.")
    return parser.parse_args()


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main(
    inventory_json: Path | str = "nodes_inventory.json",
    clash_pairs_csv: Path | str = "major_clash_pairs.csv",
    output_csv: Path | str = "clash_probability_by_pkg.csv",
    n_packs: int = 2,
    top_k: int = 15,
    precision: int = 2,
):
    """
    Calculates the probability of encountering a major version clash per package,
    weighted by downloads, assuming a user installs n_packs.

    Args:
        inventory_json: Path to the node inventory JSON file.
        clash_pairs_csv: Path to the clash pairs CSV file.
        output_csv: Path for the output CSV file with probabilities.
        n_packs: Assumed number of packs a user installs.
        top_k: Number of top packages to print to the console.
        precision: Decimal precision for console output percentages.
    """
    if not _PANDAS_AVAILABLE:
        print("Error: pandas library is required for this script.", file=sys.stderr)
        print("       Install it with: pip install pandas", file=sys.stderr)
        return # Or raise ImportError

    # Ensure inputs are Path objects
    INVENTORY_JSON = Path(inventory_json)
    CLASH_PAIRS_CSV = Path(clash_pairs_csv)
    OUT_CSV = Path(output_csv)

    # Check if input files exist
    if not INVENTORY_JSON.exists():
        print(f"Error: Inventory JSON file not found at {INVENTORY_JSON}", file=sys.stderr)
        return
    if not CLASH_PAIRS_CSV.exists():
        print(f"Error: Clash pairs CSV file not found at {CLASH_PAIRS_CSV}", file=sys.stderr)
        return

    # 1 ── Load download counts ------------------------------------------- #
    downloads: Dict[str, int] = {}
    try:
        with INVENTORY_JSON.open() as fh:
            inventory_data = json.load(fh)
        # Ensure 'id' exists and handle potential missing 'downloads' or non-int values
        for item in inventory_data:
            node_id = item.get("id")
            if not node_id:
                 print(f"Warning: Skipping item with missing 'id' in {INVENTORY_JSON}", file=sys.stderr)
                 continue
            try:
                dl_count = int(item.get("downloads", 0))
                downloads[node_id] = dl_count
            except (ValueError, TypeError):
                print(f"Warning: Invalid download count for node '{node_id}' in {INVENTORY_JSON}. Treating as 0.", file=sys.stderr)
                downloads[node_id] = 0

    except FileNotFoundError:
        print(f"Error: File not found {INVENTORY_JSON}", file=sys.stderr) # Should be caught above, but belt-and-suspenders
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {INVENTORY_JSON}: {e}", file=sys.stderr)
        return
    except Exception as e:
         print(f"Error reading or processing {INVENTORY_JSON}: {e}", file=sys.stderr)
         return


    total_dl = sum(downloads.values())
    if total_dl == 0:
        print("Warning: All download counts are zero in the inventory file. Probabilities will be zero.", file=sys.stderr)
        # Decide whether to exit or continue (continuing will result in 0 probabilities)
        # return # Or sys.exit(1)

    # 2 ── Collect unique unordered clash pairs per package ---------- #
    pkg_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    try:
        with CLASH_PAIRS_CSV.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not all(f in reader.fieldnames for f in ["node_id", "package_spec", "conflicting_nodes"]):
                 print(f"Error: Missing required columns in {CLASH_PAIRS_CSV}. Expected 'node_id', 'package_spec', 'conflicting_nodes'.", file=sys.stderr)
                 return

            for row in reader:
                node_a   = row["node_id"]
                 # Safely split package_spec
                pkg_spec_parts = row["package_spec"].split(maxsplit=1)
                if len(pkg_spec_parts) == 2:
                    pkg = pkg_spec_parts[0]
                else:
                    print(f"Warning: Skipping row with unexpected package_spec format '{row['package_spec']}' in {CLASH_PAIRS_CSV}", file=sys.stderr)
                    continue # Skip this row

                for node_b, _ in parse_conflict_list(row["conflicting_nodes"]):
                    # Ensure both nodes exist in downloads map, otherwise the pair is irrelevant for probability calc
                    if node_a in downloads and node_b in downloads:
                        pair = tuple(sorted((node_a, node_b))) # Canonical ordering
                        pkg_pairs[pkg].add(pair)
                    # else: # Optional: Warn about pairs involving nodes not in snapshot
                    #    print(f"Debug: Skipping pair ({node_a}, {node_b}) for pkg {pkg} due to missing downloads data.", file=sys.stderr)

    except FileNotFoundError:
         print(f"Error: File not found {CLASH_PAIRS_CSV}", file=sys.stderr) # Should be caught above
         return
    except Exception as e:
        print(f"Error reading {CLASH_PAIRS_CSV}: {e}", file=sys.stderr)
        return


    # 3 ── Compute probabilities p_pair and p_n ------------------------ #
    rows = []
    for pkg, pairs in pkg_pairs.items():
        num = 0
        if total_dl > 0: # Avoid division by zero if total downloads is 0
            for a, b in pairs:
                # .get() is crucial here as pair might contain nodes filtered out earlier if strict
                dl_a = downloads.get(a, 0)
                dl_b = downloads.get(b, 0)
                num += dl_a * dl_b
            p_pair = num / (total_dl * total_dl)
        else:
            p_pair = 0.0

        # Probability of at least one clash in n_packs draws (using combinations)
        p_n = 0.0
        if n_packs >= 2 and p_pair > 0: # Need at least 2 packs and non-zero pair probability
            try:
                # Calculate combinations C(n, 2)
                comb = math.comb(n_packs, 2) # Requires Python 3.8+
                # Alternative for older Python: comb = n_packs * (n_packs - 1) // 2
                # Use log-sum-exp trick for numerical stability if comb is large?
                # For typical n_packs (e.g., < 100), direct power should be fine.
                p_n  = 1.0 - (1.0 - p_pair) ** comb
            except OverflowError:
                 print(f"Warning: Overflow calculating combinations or power for pkg '{pkg}' with n_packs={n_packs}. Setting p_n to 1.0.", file=sys.stderr)
                 p_n = 1.0 # If combinations or power overflows, probability is effectively 1
            except ValueError: # e.g. if p_pair somehow became > 1, unlikely here
                 print(f"Warning: Calculation error for pkg '{pkg}'. Setting p_n to 0.0.", file=sys.stderr)
                 p_n = 0.0


        rows.append(
            {
                "package": pkg,
                "p_pair": p_pair,       # Probability a random pair clashes
                "p_n": p_n,             # Probability >=1 clash in n_packs installs
                "percent_n": p_n * 100, # p_n as percentage
            }
        )

    if not rows:
        print("No clash data processed or no packages found with valid pairs. Output CSV will be empty.", file=sys.stderr)
        # Create empty CSV with headers?
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="") as fh:
             writer = csv.writer(fh)
             writer.writerow(["package", "p_pair", "p_n", "percent_n"]) # Write header
        print(f"✓ wrote empty {OUT_CSV}\n")
        return


    df = (
        pd.DataFrame(rows)
          .sort_values("p_n", ascending=False)
          .reset_index(drop=True)
    )

    try:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        df.to_csv(OUT_CSV, index=False, float_format=f'%.{max(8, precision+2)}g') # Use sufficient precision for CSV
        print(f"✓ wrote {OUT_CSV}\n")
    except Exception as e:
        print(f"Error writing CSV to {OUT_CSV}: {e}", file=sys.stderr)


    # 4 ── Print top-k results to console ---------------------------------- #
    if not df.empty:
        print(f"--- Top {min(top_k, len(df))} packages by clash probability (assuming n_packs = {n_packs}) ---")
        # Dynamically determine padding for package name based on top_k
        max_pkg_len = df['package'].head(top_k).str.len().max() if top_k > 0 else 20
        header_fmt = f"{{:<{max_pkg_len}}}   {{}}"
        row_fmt    = f"{{:<{max_pkg_len}}}   {{:.{precision}f}} %"
        print(header_fmt.format("Package", f"P(≥1 clash in {n_packs})"))
        print("-" * (max_pkg_len + 3 + 10 + len(str(n_packs)))) # Dynamic separator

        for _, r in df.head(top_k).iterrows():
            # Ensure package name isn't longer than calculated max_len for alignment
            pkg_name_display = r.package[:max_pkg_len]
            print(row_fmt.format(pkg_name_display, r.percent_n))
    else:
        print("No results to display in console.")


# ------------------------------------------------------------------ #
# Command-Line Interface
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    args = parse_arguments()
    if not _PANDAS_AVAILABLE:
         # Check moved here so CLI fails early if pandas not available
         print("Error: pandas library is required for this script.", file=sys.stderr)
         print("       Install it with: pip install pandas", file=sys.stderr)
         sys.exit(1)

    main(
        inventory_json=args.inventory_json,
        clash_pairs_csv=args.clash_pairs_csv,
        output_csv=args.output_csv,
        n_packs=args.n,
        top_k=args.top_k,
        precision=args.prec,
    )
