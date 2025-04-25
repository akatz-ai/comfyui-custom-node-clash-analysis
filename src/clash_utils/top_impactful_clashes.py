#!/usr/bin/env python3
"""
Rank the 'loudest' strict-pin clashes by download impact
and put the more-popular pack first in every pair.

Inputs
------
strict_packs_with_major_clash.csv   node_id,downloads
major_clash_pairs.csv               node_id,package_spec,conflicting_nodes

Outputs
-------
top_impactful_clashes.csv
top_impactful_clashes.html          (interactive Plotly bar chart)

Example run
-----------
uv run top_impactful_clashes.py --downloads-csv strict_packs_with_major_clash.csv --clash-pairs-csv major_clash_pairs.csv --top-n 50
"""

from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make Plotly and Pandas optional for easier import if just using parts
try:
    import pandas as pd
    import plotly.express as px
    _PANDAS_AVAILABLE = True
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    _PLOTLY_AVAILABLE = False


# --------------------------------------------------------------------- #
# helper:  "A:spec ; B:spec2"  →  [(A,spec), (B,spec2)]
# --------------------------------------------------------------------- #
def parse_conflict_list(raw: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for piece in raw.split(";"):
        piece = piece.strip()
        if piece:
            # Handle potential missing colon gracefully, though input expects it
            if ":" in piece:
                node, spec = piece.split(":", 1)
                out.append((node.strip(), spec.strip()))
            else:
                # Or log a warning: print(f"Warning: Malformed conflict piece '{piece}'", file=sys.stderr)
                pass # Silently ignore malformed pieces for now
    return out

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Rank strict-pin clashes by download impact.")
    parser.add_argument("--downloads-csv", type=Path, default="strict_packs_with_major_clash.csv",
                        help="Input CSV file mapping node_id to downloads.")
    parser.add_argument("--clash-pairs-csv", type=Path, default="major_clash_pairs.csv",
                        help="Input CSV file with node clash pairs.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top clashes to report.")
    parser.add_argument("--output-csv", type=Path, default="top_impactful_clashes.csv",
                        help="Output CSV file path.")
    parser.add_argument("--output-html", type=Path, default="top_impactful_clashes.html",
                        help="Output HTML file path for the plot.")
    return parser.parse_args()


def main(
    downloads_csv: Path | str = "strict_packs_with_major_clash.csv",
    clash_pairs_csv: Path | str = "major_clash_pairs.csv",
    top_n: int = 20,
    output_csv: Path | str = "top_impactful_clashes.csv",
    output_html: Path | str = "top_impactful_clashes.html",
    create_plot: bool = True, # Add option to skip plot creation
):
    """
    Ranks strict-pin clashes by download impact and outputs CSV and optionally an HTML plot.

    Args:
        downloads_csv: Path to the input CSV with node downloads.
        clash_pairs_csv: Path to the input CSV with clash pairs.
        top_n: Number of top clashes to include in the output.
        output_csv: Path for the output CSV file.
        output_html: Path for the output HTML plot file.
        create_plot: Whether to generate the Plotly HTML chart.
    """
    if create_plot and not (_PANDAS_AVAILABLE and _PLOTLY_AVAILABLE):
        print("Warning: pandas and plotly are required to create the plot. Skipping plot generation.", file=sys.stderr)
        print("         Install them with: pip install pandas plotly", file=sys.stderr)
        create_plot = False

    # Ensure inputs are Path objects
    DOWNLOADS_CSV = Path(downloads_csv)
    CLASH_PAIRS_CSV = Path(clash_pairs_csv)
    OUT_CSV = Path(output_csv)
    OUT_HTML = Path(output_html)

    # Check if input files exist
    if not DOWNLOADS_CSV.exists():
        print(f"Error: Downloads CSV file not found at {DOWNLOADS_CSV}", file=sys.stderr)
        return
    if not CLASH_PAIRS_CSV.exists():
        print(f"Error: Clash pairs CSV file not found at {CLASH_PAIRS_CSV}", file=sys.stderr)
        return

    # ------------------------------------------------------------------ #
    # 1.  Load downloads mapping
    # ------------------------------------------------------------------ #
    downloads: Dict[str, int] = {}
    try:
        with DOWNLOADS_CSV.open(newline='') as fh:
            rdr = csv.reader(fh)
            header = next(rdr) # Read header
            # Find column indices robustly
            try:
                node_col_idx = header.index('node_id')
                dl_col_idx = header.index('downloads')
            except ValueError as e:
                print(f"Error: Missing required column in {DOWNLOADS_CSV}: {e}", file=sys.stderr)
                return

            for row in rdr:
                 if len(row) > max(node_col_idx, dl_col_idx): # Basic check for row length
                    node = row[node_col_idx]
                    try:
                        cnt = int(row[dl_col_idx])
                        downloads[node] = cnt
                    except ValueError:
                        # Handle cases where download count isn't a valid integer
                        print(f"Warning: Invalid download count for node '{node}' in {DOWNLOADS_CSV}. Treating as 0.", file=sys.stderr)
                        downloads[node] = 0
                 else:
                     print(f"Warning: Skipping short row in {DOWNLOADS_CSV}: {row}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File not found {DOWNLOADS_CSV}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading {DOWNLOADS_CSV}: {e}", file=sys.stderr)
        return

    # ------------------------------------------------------------------ #
    # 2.  Process clash pairs and determine order
    # ------------------------------------------------------------------ #
    rows = []
    processed_pairs = set() # To avoid duplicates like (A, B, pkg) and (B, A, pkg)

    try:
        with CLASH_PAIRS_CSV.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not all(f in reader.fieldnames for f in ["node_id", "package_spec", "conflicting_nodes"]):
                 print(f"Error: Missing required columns in {CLASH_PAIRS_CSV}. Expected 'node_id', 'package_spec', 'conflicting_nodes'.", file=sys.stderr)
                 return

            for row in reader:
                node_a        = row["node_id"]
                pkg_spec_a    = row["package_spec"]               # "torch torch==2.4.1"

                # Safely split package_spec
                pkg_spec_parts = pkg_spec_a.split(maxsplit=1)
                if len(pkg_spec_parts) == 2:
                    pkg, spec_a = pkg_spec_parts
                else:
                    # Handle unexpected format
                    print(f"Warning: Skipping row with unexpected package_spec format '{pkg_spec_a}' in {CLASH_PAIRS_CSV}", file=sys.stderr)
                    continue # Skip this row

                dl_a          = downloads.get(node_a, 0)

                for node_b, spec_b in parse_conflict_list(row["conflicting_nodes"]):
                    dl_b = downloads.get(node_b, 0)

                    # Create a canonical representation for the pair to avoid duplicates
                    pair_key = tuple(sorted((node_a, node_b))) + (pkg,)
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)


                    # decide ordering
                    if (dl_a >  dl_b) or (dl_a == dl_b and node_a < node_b):
                        main_pack, main_dl, main_spec = node_a, dl_a, spec_a
                        oth_pack,  oth_dl,  oth_spec  = node_b, dl_b, spec_b
                    else:
                        main_pack, main_dl, main_spec = node_b, dl_b, spec_b
                        oth_pack,  oth_dl,  oth_spec  = node_a, dl_a, spec_a

                    rows.append(
                        dict(
                            pack_main      = main_pack,
                            pack_other     = oth_pack,
                            package        = pkg,
                            spec_main      = main_spec,
                            spec_other     = oth_spec,
                            downloads_main = main_dl,
                            downloads_other= oth_dl,
                            impact         = main_dl + oth_dl,          # Simple sum impact
                        )
                    )
    except FileNotFoundError:
        print(f"Error: File not found {CLASH_PAIRS_CSV}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading {CLASH_PAIRS_CSV}: {e}", file=sys.stderr)
        return

    if not rows:
        print("No clash data processed. Exiting.", file=sys.stderr)
        return

    if not _PANDAS_AVAILABLE:
         print("Error: pandas library is required for sorting and output.", file=sys.stderr)
         print("       Install it with: pip install pandas", file=sys.stderr)
         return


    df = pd.DataFrame(rows)
    # No need for drop_duplicates if processed_pairs logic is correct

    # ------------------------------------------------------------------ #
    # 3.  Rank, save CSV
    # ------------------------------------------------------------------ #
    df_sorted = (
        df.sort_values(["impact", "downloads_main"], ascending=False) # Primary sort by impact
          .head(top_n)
          .reset_index(drop=True)
    )

    try:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        df_sorted.to_csv(OUT_CSV, index=False)
        print(f"✓ wrote {OUT_CSV}")
    except Exception as e:
         print(f"Error writing CSV to {OUT_CSV}: {e}", file=sys.stderr)


    # ------------------------------------------------------------------ #
    # 4.  Optionally create Plotly bar chart
    # ------------------------------------------------------------------ #
    if create_plot and _PLOTLY_AVAILABLE and not df_sorted.empty:
        # Check if required columns exist for plotting, prevent errors on empty df
        required_plot_cols = ['pack_main', 'downloads_main', 'pack_other', 'downloads_other', 'package', 'spec_main', 'spec_other', 'impact']
        if not all(col in df_sorted.columns for col in required_plot_cols):
            print(f"Warning: Missing columns required for plotting in the sorted data. Skipping plot.", file=sys.stderr)

        else:
            try:
                # Construct labels carefully, handling potential missing data if needed
                labels = (
                    df_sorted["pack_main"].astype(str)  + " (" + df_sorted["downloads_main"].astype(str)  + ")  ×  "
                  + df_sorted["pack_other"].astype(str) + " (" + df_sorted["downloads_other"].astype(str) + ") | "
                  + df_sorted["package"].astype(str) + ": " + df_sorted["spec_main"].astype(str)
                  + " vs " + df_sorted["spec_other"].astype(str)
                )

                fig = px.bar(
                    df_sorted,
                    x="impact",
                    y=labels,
                    orientation="h",
                    height=max(600, len(df_sorted) * 35), # Adjust height dynamically
                    labels={"impact": "Combined Downloads (Impact)", "y": "Clashing Pair & Package"}, # More descriptive label
                    title=f"Top {min(top_n, len(df_sorted))} Strict-Pin Clashes by Combined Downloads",
                )
                fig.update_layout(yaxis_title="", yaxis=dict(autorange="reversed")) # Keep newest/topmost at top

                OUT_HTML.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                fig.write_html(OUT_HTML)
                print(f"✓ interactive chart: {OUT_HTML}")
            except Exception as e:
                print(f"Error creating or writing HTML plot to {OUT_HTML}: {e}", file=sys.stderr)


if __name__ == "__main__":
    args = parse_arguments()
    main(
        downloads_csv=args.downloads_csv,
        clash_pairs_csv=args.clash_pairs_csv,
        top_n=args.top_n,
        output_csv=args.output_csv,
        output_html=args.output_html,
        # create_plot=True # Default is True, could be controlled by another arg if needed
    )
