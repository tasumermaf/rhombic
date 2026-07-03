#!/usr/bin/env python
"""Materialize the full-text PDF library for the research suite.

Downloads every arXiv paper in research/manifest.json to research/pdf/
(gitignored — arXiv's non-exclusive license does not permit wholesale
redistribution, so PDFs live locally; the committed cards carry metadata,
abstracts, and the program's assessments).

Usage:  python research/tools/fetch_papers.py [--delay 3]
Idempotent: existing files are skipped. Non-arXiv (DOI-only) entries are
listed at the end for manual retrieval.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=3.0, help="Seconds between downloads (arXiv rate courtesy)")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    outdir = ROOT / "pdf"
    outdir.mkdir(exist_ok=True)

    skipped_doi, failed, got = [], [], 0
    todo = [m for m in manifest if m.get("arxiv_id")]
    for i, m in enumerate(todo, 1):
        aid = m["arxiv_id"]
        dest = outdir / f"{aid.replace('/', '-')}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        url = f"https://arxiv.org/pdf/{aid}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tasumermaf-research-suite/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            got += 1
            print(f"[{i}/{len(todo)}] {aid} -> {dest.name} ({dest.stat().st_size // 1024} KB)")
        except Exception as e:  # noqa: BLE001
            failed.append((aid, str(e)[:80]))
            print(f"[{i}/{len(todo)}] {aid} FAILED: {e}")
        time.sleep(args.delay)

    for m in manifest:
        if not m.get("arxiv_id") and m.get("doi"):
            skipped_doi.append(m["doi"])

    print(f"\nDone: {got} downloaded, {len(failed)} failed, "
          f"{len([m for m in todo if (outdir / (m['arxiv_id'].replace('/','-') + '.pdf')).exists()])} total on disk.")
    if skipped_doi:
        print("DOI-only entries (retrieve manually):")
        for d in skipped_doi:
            print(f"  https://doi.org/{d}")
    if failed:
        print("Failed (retry later):", ", ".join(a for a, _ in failed))


if __name__ == "__main__":
    main()
