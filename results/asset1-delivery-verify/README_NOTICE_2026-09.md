# DATED NOTICE (2026-09) — Asset-1 delivery-verification bundle: byte representations

**Filed 2026-09-30 by the decider (Meridian)**, on the Director's ruling of
2026-09-29 (handover-2 rulings, item (t), `DIRECTOR_RULINGS_HANDOVER2_2026-09-29`,
not filed in this repository). This file is the dated sibling of
`README.md` and `SHA256SUMS` in this directory. **It does not edit either
file.**

The README directs a reader to verify this bundle with `SHA256SUMS`. Errata §1
(`docs/ERRATA_2026-09-18.md`, "XR-001 and Asset-1 byte representations")
records what that check meets on a checkout: four of the 14 available text
files match their recorded checksum directly, while ten match only after exact
LF-to-CRLF reconstruction, and the two reduced feature NPZ files are
intentionally supplied out of band, as the README states, so a checkout does
not verify them. The explicit byte representations, each with its historical
checksum source, are registered in
`docs/representation-receipts/asset1-2026-09-18.json` (the XR-001 counterpart
is `docs/representation-receipts/xr001-2026-09-18.json`), and errata §1 gives
the read-only command that checks them, under which a public checkout is
expected to return exit code 2 (incomplete coverage) for this bundle until its
two original feature matrices are provided. The README's own historical
checksum, `SHA256SUMS` line 16, is its LF form,
`1a124cb4c4293eb25093d6dae466f2aa9e90966d171e19919f26aa69b7fb0a01`; the
receipt registers its CRLF form as
`219510aed7d2ff887b9b46d00fc28baec170c863f2908dd62b985d2e548085ec`. This
notice changes no result, no checksum and no frozen file; it records where the
correction for this directory lives.
