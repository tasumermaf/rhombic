#!/usr/bin/env python3
"""Verify explicitly registered LF/CRLF representations without changing evidence.

This verifies bytes and the cited frozen checksum fields, not model results.
It does not rewrite manifests, relax the XR-001 runner, or execute a run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("receipt path must stay inside the repository")
    return path


def frozen_checksum(root: Path, source: dict) -> str:
    path = within(root, source["path"])
    if source["kind"] == "json-field":
        return json.loads(path.read_text(encoding="utf-8"))[source["field"]]
    if source["kind"] == "sha256sums":
        matches = []
        for line in path.read_text(encoding="utf-8").splitlines():
            checksum, name = line.split(maxsplit=1)
            if name.lstrip("*") == source["entry"]:
                matches.append(checksum)
        if len(matches) != 1:
            raise ValueError("frozen checksum entry is missing or ambiguous")
        return matches[0]
    raise ValueError("unsupported frozen checksum source")


def verify(root: Path, receipt: dict) -> dict:
    root = root.resolve()
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported representation receipt schema")
    results = []
    paths = set()
    for entry in receipt["entries"]:
        name = entry["path"]
        result = {"path": name}
        try:
            if name in paths:
                raise ValueError("duplicate representation entry")
            paths.add(name)
            path = within(root, name)
            historical = frozen_checksum(root, entry["checksum_source"])
            if historical != entry["historical_sha256"]:
                raise ValueError("frozen checksum differs from reviewed receipt")
            if not path.is_file():
                result.update(status="unavailable", reason="artifact absent; not verified")
            else:
                raw = path.read_bytes()
                digest = sha256(raw)
                representations = entry["representations"]
                if digest not in representations.values():
                    raise ValueError("bytes match no explicitly reviewed representation")
                if entry["transform"] == "crlf-to-lf":
                    lf = raw.replace(b"\r\n", b"\n")
                    if b"\r" in lf:
                        raise ValueError("standalone carriage return is not a reviewed EOL")
                    derived = {"lf": sha256(lf), "crlf": sha256(lf.replace(b"\n", b"\r\n"))}
                    if representations != derived or entry["lf_sha256"] != derived["lf"]:
                        raise ValueError("declared CRLF-to-LF transformation does not match")
                    if historical not in derived.values():
                        raise ValueError("neither exact EOL representation matches the historical checksum")
                elif entry["transform"] == "identity":
                    if digest != historical or set(representations.values()) != {historical}:
                        raise ValueError("identity representation differs from historical bytes")
                else:
                    raise ValueError("unsupported representation transformation")
                result.update(status="verified", actual_sha256=digest,
                              representation=next(k for k,v in representations.items() if v == digest),
                              historical_sha256=historical,
                              byte_identical_to_historical=(digest == historical))
        except (KeyError, OSError, ValueError, TypeError) as exc:
            result.update(status="failed", reason=str(exc))
        results.append(result)
    if not results:
        raise ValueError("receipt has no artifact entries")
    status = ("failed" if any(r["status"] == "failed" for r in results) else
              "incomplete" if any(r["status"] == "unavailable" for r in results) else "verified")
    return {"schema_version": 1, "status": status,
            "scope": "Registered byte representations and cited checksum fields only; no result replication",
            "entries": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.root, json.loads(args.receipt.read_text(encoding="utf-8")))
    except (KeyError, OSError, ValueError, TypeError) as exc:
        report = {"status": "failed", "error": str(exc)}
    print(json.dumps(report, indent=2))
    return {"verified": 0, "failed": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
