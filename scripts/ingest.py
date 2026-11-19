#!/usr/bin/env python
"""Build the processed sample in data/sample/.

    python scripts/ingest.py                 # real data if data/raw/ has it, else synthetic
    python scripts/ingest.py --source kaggle # fail loudly if the raw files are absent
    python scripts/ingest.py --source synthetic

Writes four Parquet tables plus provenance.json, then prints what it built.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/ingest.py` from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supplyiq import ingest, paths  # noqa: E402
from supplyiq.config import load_assumptions  # noqa: E402
from supplyiq.data import write_provenance  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the SupplyIQ processed sample.")
    ap.add_argument("--source", choices=["auto", "kaggle", "synthetic"], default="auto")
    ap.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=paths.SAMPLE_DIR)
    args = ap.parse_args(argv)

    a = load_assumptions()
    found = ingest.kaggle_files_present(args.raw_dir)
    print(f"raw dir: {args.raw_dir}")
    for key, path in found.items():
        print(f"  {key:9s} {'FOUND  ' + str(path.name) if path else 'missing'}")

    if args.source == "kaggle" and not ingest.have_kaggle(args.raw_dir):
        print(
            "\nERROR: --source kaggle was requested but the raw files are not all present.\n"
            "       See README > Data for the download steps, or use --source synthetic.",
            file=sys.stderr,
        )
        return 2

    print(f"\nbuilding (source={args.source}) ...")
    tables, prov = ingest.build(prefer=args.source, raw_dir=args.raw_dir, a=a)
    written = ingest.write_sample(tables, args.out_dir)
    prov_path = write_provenance(prov, args.out_dir / paths.PROVENANCE_FILE.name)

    print(f"\norigin: {prov.origin.upper()}  ({prov.badge})")
    total_bytes = 0
    for name, path in written.items():
        size = path.stat().st_size
        total_bytes += size
        print(f"  {name:16s} {len(tables[name]):>8,} rows  {size / 1e6:6.2f} MB  {path.name}")
    print(f"  {'provenance':16s} {'':>8}       "
          f"{prov_path.stat().st_size / 1e6:6.2f} MB  {prov_path.name}")
    print(f"\ntotal committed sample: {total_bytes / 1e6:.2f} MB (budget 25 MB)")
    if total_bytes > 25e6:
        print("WARNING: sample exceeds the 25 MB budget for the committed repo.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
