# History rewrite of 2026-09-11

On 2026-09-11 the full history of this repository was rewritten (computed on 2026-09-08 from the history at `df22b69`, published on 2026-09-11 at 2026-09-11 14:58 UTC). Earlier commits carried, inline in a few source files, result records, rendered figures and one commit message, research data that is proprietary to the program and is described in [docs/IP_BOUNDARY_SPEC.md](IP_BOUNDARY_SPEC.md) (no credentials were involved). The current tree had already been cleared of that material before the rewrite; the rewrite carries the same redaction back through every earlier commit, on every branch and tag.

## The rule

Every historical version of a file that carried protected material was replaced by the version of that path at the pre-rewrite `main` HEAD; clean historical versions of the same paths were left in place. The one affected path that exists only on `gh-pages` (the site's copy of the weave banner) was replaced by the `main` HEAD blob of `assets/weave_banner.png`, which its historical versions were copies of. For the weave renders, their generating script, and the paths marked *all versions* below, every historical version was replaced, because a rendered image or a structurally redacted script carries no token a text scan can flag. Nothing else changed: commit topology, authors, committers and dates all survive; messages survive except that an abbreviated commit SHA quoted inside a message was updated to the new SHA of the commit it names, and one message had a protected name and value replaced by a withheld marker. Because the first affected commit is early in the history, the SHA of essentially every commit changed. Historical commits therefore show a few anachronistic files (a March commit carrying a September redaction notice); that is disclosed here, not hidden. Commits whose only changes were inside replaced files became empty and were dropped; they are listed below with the nearest surviving successor.

Blob substitution was chosen over path deletion because the paths are load-bearing (paper sources, result records, the corpus module): deleting them would break every historical build, substituting the redacted version keeps history buildable and removes exactly the protected bytes.

## What the rewrite did

| quantity | value |
|---|---|
| blobs substituted | 48 |
| paths whose history changed | 22 |
| commit messages redacted | 1 |
| commits before / after | 196 / 190 |
| commits rewritten (new SHA) | 170 |
| commits unchanged | 20 |
| commits dropped as empty | 6 |
| `main` HEAD before / after | `df22b69` / `9c1c7d8` |
| `main` HEAD tree identical | True |
| historical blobs still flagged after the rewrite, other than the current reviewed HEAD versions | 0 |
| commit or tag messages still flagged after the rewrite | 0 |
| `main` commits whose message quoted another commit's SHA and was updated to the new SHA | 19 |
| `main` commits whose author, committer or dates changed | 0 |

Paths whose history changed (*all versions* = every historical version now equals the pre-rewrite HEAD version; otherwise only the flagged versions were replaced; *gh-pages* = a path that exists only on the `gh-pages` branch):

- `assets/weave_banner.png` — *all versions*
- `assets/weave_pattern.png` — *all versions*
- `paper/audit/round-1/agent-1B-math-p2.md` — *all versions*
- `paper/audit/round-1/agent-1D-ip-boundary.md`
- `paper/audit/round-3/agent-3A-narrative.md` — *all versions*
- `paper/audit/round-3/agent-3C-crossrefs.md` — *all versions*
- `paper/audit/round-7/agent-7B-cross-paper.md` — *all versions*
- `paper/figures/fig5-spectral-stems.png` — *all versions*
- `paper/rhombic-paper2.tex` — *all versions*
- `paper/rhombic-paper2_IT.tex` — *all versions*
- `results/paper2/INTERPRETATION.md` — *all versions*
- `results/paper2/RESULTS.md` — *all versions*
- `results/paper2/experiment_3_prime_coherence.txt` — *all versions*
- `results/paper2/experiment_6_prime_vertex.txt`
- `rhombic/corpus.py`
- `scripts/generate_harmony.py` — *all versions*
- `scripts/generate_weave.py` — *all versions*
- `scripts/render_ep2_slides.py` — *all versions*
- `space/app.py`
- `tests/test_corpus.py` — *all versions*
- `weave_banner.png` — *gh-pages*
- `website/weave_banner.png` — *all versions*

## Branches and tags

| ref | before | after |
|---|---|---|
| `dependabot/github_actions/actions/checkout-7` | `2f95682` | `9ab0538` |
| `dependabot/github_actions/actions/configure-pages-6` | `b882394` | `c41696a` |
| `dependabot/github_actions/actions/deploy-pages-5` | `c7b24fc` | `1803e08` |
| `dependabot/github_actions/actions/setup-python-7` | `416f7a8` | `7df7a37` |
| `dependabot/github_actions/actions/upload-pages-artifact-5` | `0794adc` | `6fb75c3` |
| `gh-pages` | `8bade96` | `2a177c4` |
| `main` | `df22b69` | `9c1c7d8` |
| tag `asset1-bundle-anchor` | `638f4a8` | `677766f` |
| tag `v0.1.0` | `37ccdfb` | `37ccdfb` |
| tag `v0.1.1` | `5c8ae86` | `5c8ae86` |
| tag `v0.1.2` | `e1d07ba` | `e1d07ba` |
| tag `v0.2.0` | `b345517` | `d40e3b4` |
| tag `v0.3.0` | `c4d2845` | `ee6b96d` |

Tags keep their names. Released material that cites a tag (`asset1-bundle-anchor` in the Asset-1 paper) still resolves by that name; an archive SHA-256 printed in a paper is a digest of a file, not a commit, and is unaffected.

## Commits cited in released material

Old SHA as printed, new SHA to use. The map published here covers every commit cited in this repository's documents, in released material, at a branch or tag tip, or dropped as empty (74 pairs): [docs/history-rewrite/cited-commit-map.txt](history-rewrite/cited-commit-map.txt) (one `old new` pair per line; an all-zero new SHA marks a dropped commit). The per-document map, Director memos first, is [docs/history-rewrite/COMMIT_MAP_BY_DOCUMENT.md](history-rewrite/COMMIT_MAP_BY_DOCUMENT.md). The maintainers hold the complete map privately and will resolve any other pre-rewrite SHA on request.

| cited | now | status | date and subject |
|---|---|---|---|
| `03f0402` | `db79be3` | mapped | 2026-08-04 S2 gate satisfied: 9/9 accessible timing runs meas |
| `0f39f77` | `3961e52` | mapped | 2026-07-29 Tinker signal pilot (E-T4 pre-step): SIGNAL=MIXED, |
| `16605d1` | `c65e2c2` | mapped | 2026-08-04 H2-S amendment v2: rank-56 arm withdrawn (unbuilda |
| `2e1f823` | `6bf282d` | mapped | 2026-07-21 Next-round designs: H2 scale-up prereg draft, gran |
| `377d351` | `b1005fe` | mapped | 2026-07-21 Asset-1 paper: LaTeX conversion — rhombic-asset1.t |
| `4d2224a` | `d12be62` | mapped | 2026-03-06 Add dependency diagram, fix figure titles, align c |
| `638f4a8` | `677766f` | mapped | 2026-07-21 Asset-1 delivery-verification bundle for the Direc |
| `69bb6b0` | `0b470f7` | mapped | 2026-07-07 prereg: Director verified all A3-A5 condition enco |
| `7c136ad` | `9ed5b66` | mapped | 2026-08-04 fcc_diameter_check FIXED + reproduced; decider rul |
| `7e747b7` | `7186de7` | mapped | 2026-07-21 Asset-1 paper: citation-verification fixes (R4 pas |
| `8c16ecf` | `8f5eb32` | mapped | 2026-07-07 docs: front matter refreshed to reflect the July l |
| `8fa3d77` | `c09c0c6` | mapped | 2026-08-04 granularity: Stage-0 label spaces frozen (L0/L1/Ar |
| `99e0ac5` | `95deca5` | mapped | 2026-07-03 Round 2: adopt Director's review obligations |
| `a556588` | `61fa637` | mapped | 2026-07-07 Workspace-mapping lanes E-1/E-2/E-3: output-refere |
| `a703cc2` | `2a9afb9` | mapped | 2026-07-08 gitignore: T-001r3 local checkpoint payload (r2 co |
| `a777adb` | `0725c50` | mapped | 2026-07-10 XR-001 RESULTS: H-compaction CONFIRMED — structure |
| `a8fe241` | `f2c1ca0` | mapped | 2026-07-07 Director pinnings ruling (2026-07-07): two output  |
| `bae4079` | `5ed2c7f` | mapped | 2026-07-10 XR-001: episode-count guard tolerates legacy manif |
| `ccf7e06` | `71540cc` | mapped | 2026-07-09 jlens: synthetic positive control PASSES - Directo |
| `df8881f` | `b034027` | mapped | 2026-07-14 website: homepage reflects the current program — d |
| `e2aa6fc` | `7651d58` | mapped | 2026-07-10 XR-001: externalization-robustness pilot — pre-reg |
| `e344477` | `f691239` | mapped | 2026-07-07 prereg: A3-A5 Director conditions encoded (ruling  |
| `fa1c4f0` | `ed947a8` | mapped | 2026-08-04 Director rulings 2026-08-04 executed: rate extract |

## Commits dropped as empty

Their only changes were inside replaced files; their messages are preserved here.

| old SHA | date | subject | nearest surviving successor |
|---|---|---|---|
| `709ac61` | 2026-03-06 | Remove residual 'universal' language (3 instances) | `d12be62` |
| `e5a4d67` | 2026-03-07 | Remove corpus values table from Paper 2 (proprietary IP) | `1138e39` |
| `d6c753c` | 2026-03-08 | Add RhombiLoRA tab to HuggingFace Space | `a23a3a2` |
| `2281148` | 2026-03-08 | Update HF Space: Paper 3 "draft" → complete | `c0782f8` |
| `04aee46` | 2026-03-09 | Replace demo weave with real tessitura prime-thread data | `6554b27` |
| `cd12a97` | 2026-03-09 | Replace demo weave with real tessitura prime-thread data | `c9ff2bf` |

## For readers

- Cite tags, not commit SHAs, from now on.
- A commit SHA printed in any document dated before this rewrite resolves only through the map above.
- The pre-rewrite history is preserved privately by the maintainers as the provenance record for every SHA ever cited.
- A push cannot rewrite GitHub's read-only pull-request refs (`refs/pull/*`), which keep most of the pre-rewrite objects reachable; GitHub has been asked to remove those refs and the now-unreachable objects. Until that completes, an old SHA may still be fetchable by direct URL or through a pull-request ref.

