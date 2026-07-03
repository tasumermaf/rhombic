#!/usr/bin/env python
"""Build the research documentation suite from extracted assessment JSON.

Pipeline:
  1. Load per-source extraction JSONs (papers + non-paper resources).
  2. Deduplicate (arXiv id > DOI > normalized title), merging assessments.
  3. Enrich arXiv entries with canonical metadata + abstracts via the arXiv API.
  4. Emit: research/data/papers.json (merged corpus, committed),
           research/papers/<slug>.md (one card per paper),
           research/INDEX.md (master table, grouped),
           research/bibliography.bib,
           research/manifest.json (for fetch_papers.py).

Usage:  python research/tools/build_suite.py --extract-dir <dir with *.json>
Re-runs are idempotent; cards are regenerated from data/papers.json + API cache.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # research/
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

GROUP_ORDER = [
    ("bridge-lineage", "Bridge-Matrix Lineage (between A and B)"),
    ("block-diagonal", "Block-Diagonal Structure"),
    ("spectral", "Spectral Methods & Dynamics"),
    ("geometric", "Geometric / Manifold / Polytope Methods"),
    ("equivariance", "Equivariance & Physics-Matched Priors"),
    ("representation-alignment", "Representation Alignment & Cross-Modal"),
    ("merging", "Merging, Routing & Task Arithmetic"),
    ("fingerprinting", "Fingerprinting, Provenance & Diagnostics"),
    ("diffusion-lora", "Diffusion / Video LoRA"),
    ("nemotron-engineering", "Engineering References (Nemotron era)"),
    ("survey", "Surveys"),
    ("director-map", "Director's Map (Representation & Adjacent)"),
    ("paper-bibliography", "Cited in Our Papers"),
    ("peripheral", "Peripheral / Monitored"),
]


def norm_title(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def norm_arxiv(aid):
    if not aid:
        return None
    aid = str(aid).strip()
    aid = re.sub(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf|html)/", "", aid)
    aid = re.sub(r"v\d+$", "", aid)
    aid = aid.replace(".pdf", "").strip("/")
    return aid or None


def slug_for(entry):
    if entry.get("arxiv_id"):
        return "arxiv-" + entry["arxiv_id"].replace("/", "-").replace(".", "-")
    base = re.sub(r"[^a-z0-9]+", "-", (entry.get("title") or "untitled").lower()).strip("-")
    return base[:60] or "untitled"


def load_extractions(extract_dir):
    papers, resources = [], []
    for f in sorted(Path(extract_dir).glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"WARN: cannot parse {f.name}: {e}")
            continue
        for p in data.get("papers", []) or []:
            p["_from"] = f.stem
            papers.append(p)
        for r in data.get("non_paper_resources", []) or []:
            r["_from"] = f.stem
            resources.append(r)
    return papers, resources


def dedup(papers):
    merged = {}
    order = []
    for p in papers:
        aid = norm_arxiv(p.get("arxiv_id"))
        doi = (p.get("doi") or "").strip().lower() or None
        key = ("arxiv", aid) if aid else ("doi", doi) if doi else ("title", norm_title(p.get("title")))
        if key not in merged:
            merged[key] = {
                "title": p.get("title"),
                "arxiv_id": aid,
                "doi": p.get("doi"),
                "other_url": p.get("other_url"),
                "authors": p.get("authors"),
                "institution": p.get("institution"),
                "venue": p.get("venue"),
                "year_date": p.get("year_date"),
                "tags": [],
                "assessments": [],
            }
            order.append(key)
        m = merged[key]
        for field in ("arxiv_id", "doi", "other_url", "authors", "institution", "venue", "year_date", "title"):
            if not m.get(field) and p.get(field):
                m[field] = norm_arxiv(p[field]) if field == "arxiv_id" else p[field]
        for t in p.get("tags") or []:
            if t not in m["tags"]:
                m["tags"].append(t)
        if p.get("assessment_verbatim") or p.get("relevance_verdict"):
            m["assessments"].append({
                "source": p.get("source_doc") or p["_from"],
                "verdict": p.get("relevance_verdict"),
                "threat": p.get("threat_level"),
                "cite_in": p.get("cite_in"),
                "confidence": p.get("confidence"),
                "text": p.get("assessment_verbatim"),
            })
    return [merged[k] for k in order]


def fetch_arxiv_metadata(entries, cache_path):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    ids = [e["arxiv_id"] for e in entries if e.get("arxiv_id") and e["arxiv_id"] not in cache]
    for i in range(0, len(ids), 40):
        batch = ids[i : i + 40]
        url = "http://export.arxiv.org/api/query?id_list=" + ",".join(batch) + f"&max_results={len(batch)}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                tree = ET.fromstring(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"WARN: arXiv API batch failed: {e}")
            continue
        for entry in tree.findall(ATOM + "entry"):
            raw_id = (entry.findtext(ATOM + "id") or "").rsplit("/", 1)[-1]
            aid = norm_arxiv(raw_id)
            title = re.sub(r"\s+", " ", entry.findtext(ATOM + "title") or "").strip()
            if not aid or not title or title.lower() == "error":
                continue
            cache[aid] = {
                "title": title,
                "abstract": re.sub(r"\s+", " ", entry.findtext(ATOM + "summary") or "").strip(),
                "authors": ", ".join(a.findtext(ATOM + "name") or "" for a in entry.findall(ATOM + "author")),
                "published": (entry.findtext(ATOM + "published") or "")[:10],
                "category": (entry.find(ARXIV + "primary_category").get("term")
                             if entry.find(ARXIV + "primary_category") is not None else None),
            }
        time.sleep(3)
        print(f"  arXiv API: {min(i + 40, len(ids))}/{len(ids)} fetched")
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def primary_group(tags):
    for tag, label in GROUP_ORDER:
        if tag in (tags or []):
            return label
    return "Uncategorized"


def write_cards(entries, meta):
    outdir = ROOT / "papers"
    outdir.mkdir(exist_ok=True)
    for e in entries:
        slug = slug_for(e)
        m = meta.get(e.get("arxiv_id") or "", {})
        title = m.get("title") or e.get("title") or "Untitled"
        lines = [f"# {title}", ""]
        idline = []
        if e.get("arxiv_id"):
            idline.append(f"arXiv: [{e['arxiv_id']}](https://arxiv.org/abs/{e['arxiv_id']})")
        if e.get("doi"):
            idline.append(f"DOI: [{e['doi']}](https://doi.org/{e['doi']})")
        if e.get("other_url"):
            idline.append(f"[link]({e['other_url']})")
        if idline:
            lines.append("> " + " · ".join(idline))
        facts = []
        if m.get("authors") or e.get("authors"):
            facts.append(f"**Authors:** {m.get('authors') or e.get('authors')}")
        if e.get("institution"):
            facts.append(f"**Institution:** {e['institution']}")
        if e.get("venue"):
            facts.append(f"**Venue:** {e['venue']}")
        if m.get("published") or e.get("year_date"):
            facts.append(f"**Date:** {m.get('published') or e.get('year_date')}")
        if m.get("category"):
            facts.append(f"**Category:** {m['category']}")
        if e.get("tags"):
            facts.append(f"**Tags:** {', '.join(e['tags'])}")
        lines += ["", "  \n".join(facts), ""]
        if m.get("abstract"):
            lines += ["## Abstract", "", m["abstract"], ""]
        if e.get("assessments"):
            lines += ["## Program assessments", ""]
            for a in e["assessments"]:
                hdr = f"### From `{a['source']}`"
                bits = [b for b in [
                    f"**Verdict:** {a['verdict']}" if a.get("verdict") else None,
                    f"**Threat:** {a['threat']}" if a.get("threat") else None,
                    f"**Cite in:** {a['cite_in']}" if a.get("cite_in") else None,
                    f"**Confidence:** {a['confidence']}" if a.get("confidence") else None,
                ] if b]
                lines += [hdr, ""]
                if bits:
                    lines += [" · ".join(bits), ""]
                if a.get("text"):
                    lines += [a["text"], ""]
        (outdir / f"{slug}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_index(entries, resources, meta):
    groups = {}
    for e in entries:
        groups.setdefault(primary_group(e.get("tags")), []).append(e)
    lines = [
        "# Research Suite — Master Index",
        "",
        f"**Papers:** {len(entries)} · **Non-paper resources:** {len(resources)} · Built by `tools/build_suite.py` from `data/papers.json`.",
        "",
        "Navigation for LLMs and humans: this index → per-paper cards in `papers/` "
        "(metadata + abstract + every program assessment, verbatim) → `SYNTHESIS.md` "
        "for cross-cutting conclusions → `tools/fetch_papers.py` to materialize full-text PDFs locally.",
        "",
    ]
    for _, label in GROUP_ORDER + [("", "Uncategorized")]:
        if label not in groups:
            continue
        lines += [f"## {label}", "", "| Paper | ID | Venue/Date | Verdict | Card |", "|---|---|---|---|---|"]
        for e in sorted(groups[label], key=lambda x: (x.get("title") or "").lower()):
            m = meta.get(e.get("arxiv_id") or "", {})
            title = (m.get("title") or e.get("title") or "Untitled")[:80]
            eid = e.get("arxiv_id") or e.get("doi") or "—"
            venue = e.get("venue") or m.get("published") or e.get("year_date") or "—"
            verdict = next((a["verdict"] for a in e.get("assessments", []) if a.get("verdict")), "—")
            lines.append(f"| {title} | {eid} | {venue} | {str(verdict)[:60]} | [card](papers/{slug_for(e)}.md) |")
        lines.append("")
    if resources:
        lines += ["## Non-Paper Resources", "", "| Name | Type | URL | Source |", "|---|---|---|---|"]
        for r in resources:
            lines.append(f"| {r.get('name','—')} | {r.get('type','—')} | {r.get('url','—')} | `{r.get('source_doc') or r.get('_from','—')}` |")
        lines.append("")
    (ROOT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_bib(entries, meta):
    out = []
    for e in entries:
        m = meta.get(e.get("arxiv_id") or "", {})
        title = m.get("title") or e.get("title")
        if not title:
            continue
        key = slug_for(e).replace("arxiv-", "a")
        authors = (m.get("authors") or e.get("authors") or "Unknown").replace(", ", " and ")
        year = (m.get("published") or str(e.get("year_date") or ""))[:4] or "2026"
        if e.get("arxiv_id"):
            out.append(f"@misc{{{key},\n  title={{{title}}},\n  author={{{authors}}},\n  year={{{year}}},\n"
                       f"  eprint={{{e['arxiv_id']}}},\n  archivePrefix={{arXiv}}\n}}")
        elif e.get("doi"):
            out.append(f"@article{{{key},\n  title={{{title}}},\n  author={{{authors}}},\n  year={{{year}}},\n"
                       f"  doi={{{e['doi']}}}\n}}")
        else:
            url = e.get("other_url") or ""
            out.append(f"@misc{{{key},\n  title={{{title}}},\n  author={{{authors}}},\n  year={{{year}}},\n"
                       f"  howpublished={{{url}}}\n}}")
    (ROOT / "bibliography.bib").write_text("\n\n".join(out) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", default=None, help="Directory of extraction JSONs (first build)")
    args = ap.parse_args()

    data_path = ROOT / "data" / "papers.json"
    res_path = ROOT / "data" / "resources.json"
    (ROOT / "data").mkdir(exist_ok=True)

    if args.extract_dir:
        papers, resources = load_extractions(args.extract_dir)
        entries = dedup(papers)
        data_path.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
        res_path.write_text(json.dumps(resources, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Merged {len(papers)} raw entries -> {len(entries)} unique papers; {len(resources)} resources")
    else:
        entries = json.loads(data_path.read_text(encoding="utf-8"))
        resources = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else []

    meta = fetch_arxiv_metadata(entries, ROOT / "data" / "arxiv_metadata_cache.json")
    write_cards(entries, meta)
    write_index(entries, resources, meta)
    write_bib(entries, meta)
    manifest = [{"arxiv_id": e.get("arxiv_id"), "doi": e.get("doi"), "slug": slug_for(e)} for e in entries]
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Suite built: {len(entries)} cards, INDEX.md, bibliography.bib, manifest.json")


if __name__ == "__main__":
    sys.exit(main())
