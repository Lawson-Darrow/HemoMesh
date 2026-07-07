# HemoMesh — SPEC

**Project:** Calibrated Uncertainty & Selective Deferral for Coronary Wall-Shear-Stress / Pressure Surrogates under Distribution Shift
**Course:** MTH 5320 (Deep Learning), Summer 2026
**Status:** scoped, pre-scaffold
**Proposal:** see `PROPOSAL.md` (v2). This SPEC is the executable plan.
**Process:** spec-first — this doc is approved before code, mirroring the Biomedical-RAG-Agent workflow.

---

## 1. Summary

Build mesh-based geometric-DL surrogates that predict per-node **wall shear stress (WSS)** and **pressure** on synthetic coronary artery meshes, and — the contribution — equip them with **calibrated, spatially-resolved uncertainty** plus a **selective-deferral** protocol, evaluated under **controlled distribution shift**. The course requirement (architecture beyond a dense network) is met by mesh GNN / equivariant / transformer backbones, with a dense MLP retained as the explicit baseline to beat. The publishable delta is UQ + shift-robustness + deferral, which prior coronary-surrogate work (Suk et al. 2024; Nannini benchmark 2025) does not address.

**Course-requirement note:** the dense MLP is the mandated baseline; at least one message-passing/equivariant/transformer backbone must beat it on WSS approximation error for the project to satisfy "more advanced than a dense network."

---

## 2. Verified dataset facts (ground truth, inspected 2026-06-30)

- **Source:** `github.com/sukjulian/coronary-mesh-convolution` (code MIT; **data CC-BY 4.0**). Download link live; 2.5 GB zip pulled and inspected.
- **Cases:** 3,999 steady-flow synthetic coronary arteries — **2,000 `single` + 1,999 `bifurcating`**.
- **Per-case HDF5 group `sample_NNNN`:** `wss` (N×3, float32), `pressure` (N×1, float32), `pos` (N×3), `face` (M×3 int triangles), `inlet_idcs`. Units CGS; WSS/pressure in dyn/cm² (= 0.1 Pa).
- **Mesh size:** single ~5.9k–11.8k nodes (mean 9.5k); bifurcating ~9.8k–24.3k (mean 16.8k).
- **Layout:** `vessel-datasets/stead/{single,bifurcating}/raw/database.hdf5`, PyG `raw`/`processed` convention. Loader pattern in repo `datasets.py`; PyG conversion in `vtk_demo.py`.
- **Baselines in repo:** GEM-GCN (gauge-equivariant), a PointNet-style baseline, DiffusionNet; **pretrained weights** downloadable (harness sanity-check).
- **Not included:** pulsatile data (steady-flow only); no explicit per-case stenosis-severity label (must derive a proxy — see §5).

---

## 3. Decisions locked (defaults applied — override before M1 if desired)

1. **Targets:** per-node **WSS (vector)** primary; **pressure (scalar)** secondary. Both verified present. [Source: dataset inspection]
2. **Backbones (Axis A, 4):** dense **MLP** (per-node, floor) · **PointNet-style** baseline (repo) · **GEM-GCN** gauge-equivariant (repo, pretrained — reused, near-zero cost) · **mesh transformer** (the one new backbone; differentiates from Suk and tests the "transformer generalizes" claim through the UQ lens).
3. **UQ (Axis B, the contribution):** **split/conformal prediction** (primary) · **deep ensembles** · **MC-dropout** (cheap baseline). **Evidential = stretch.**
4. **Distribution shift:** **single → bifurcating** (topology, core) · **severity-stratified** split via a derived proxy (core) · **real-geometry VMR** qualitative stress test (stretch). **steady → pulsatile = excluded** (not in bundle).
5. **Repo:** standalone `HemoMesh` (depends on / cites the Suk dataset).
6. **Compute:** university GPU cluster; value is parallel ablation sweeps + ensembles, not raw scale (meshes are small).

---

## 4. Scope

**In:** reuse the Suk CC-BY dataset (no CFD generation) · WSS + pressure per-node regression · 4 backbones incl. dense baseline · conformal + ensembles + MC-dropout UQ · calibration + selective-deferral + distribution-shift evaluation · bootstrap CIs + multi-seed · reproducible report + slides + released code.

**Out (deliberately):** SimVascular CFD generation · pulsatile flow · clinical vFFR / diagnostic-accuracy claims · invasive-FFR validation · patient-specific cohorts as a quantitative benchmark · pressure/vFFR from the unverifiable Nannini dataset.

---

## 5. Distribution-shift protocol

- **Shift A — topology (core):** train on `single`, evaluate on `bifurcating` (and the reverse). Cleanly defined; both splits present.
- **Shift B — severity (core):** no stored severity label, so derive a per-case proxy (one of: minimum lumen cross-sectional radius from geometry; max WSS magnitude; or total pressure-drop inlet→outlet). Stratify into mild/severe; train mild, test severe. **The proxy choice is itself reported** (and sanity-checked against WSS/pressure extremes).
- **Shift C — real geometry (stretch):** a handful of VMR coronary surface meshes (geometry only); qualitative — does predicted uncertainty spike on real anatomy? No clinical metrics.

For each shift: measure accuracy drop **and** UQ behavior (does conformal coverage fall below nominal; which UQ method's uncertainty best flags the shift).

---

## 6. Evaluation metrics

- **Field accuracy:** Su et al. (2020) **approximation error** `sqrt(Σ‖Δ‖²/Σ‖ref‖²)` (matches repo `utils/metrics.py`) + per-node **cosine similarity** (WSS vector) + NMAE/RMSE. Pressure: NMAE/RMSE, R².
- **Calibration / UQ:** conformal **coverage vs nominal** (in-distribution and per-shift); **regression ECE** + reliability diagrams; **error–uncertainty correlation** / **AUSE** (sparsification).
- **Selective prediction:** **risk–coverage curves** (defer top-k% highest-uncertainty nodes/cases → retained-set error drop); deferral efficiency; spatial map (does uncertainty concentrate at stenoses/bifurcations).
- **Rigor:** bootstrap CIs on headline metrics; multi-seed training; ablations (backbone, UQ method, conformal calibration-set size, severity-proxy choice).

---

## 7. Milestones (acceptance criteria)

**M1 — Data + baseline reproduction.** Download + verify (MD5) the Suk dataset; wire `datasets.py`/`vtk_demo.py` loading into the `hemomesh` package; **reproduce a published GEM-GCN WSS baseline number** within tolerance using the pretrained weights. *Accept:* baseline approximation-error reproduces; data loads to PyG; smoke test green. *(De-risked: no CFD generation.)*

**M2 — Backbone set (Axis A).** Implement/integrate dense MLP, PointNet-style, GEM-GCN, mesh transformer; train each on `single` (WSS + pressure). *Accept:* all 4 train to completion; ≥1 geometric backbone beats the dense MLP on approximation error with non-overlapping bootstrap CIs (course requirement); results logged to `results/experiments.csv` by config hash.

**M3 — UQ methods (Axis B).** Add conformal (primary), deep ensembles, MC-dropout to the best 1–2 backbones; in-distribution calibration eval. *Accept:* conformal coverage ≈ nominal (e.g., 90±2%) in-distribution; ECE + AUSE reported with CIs.

**M4 — Selective deferral.** Risk–coverage curves; spatial uncertainty maps. *Accept:* retained-set error decreases monotonically with deferral; uncertainty concentration at stenoses/bifurcations quantified.

**M5 — Distribution-shift study (headline).** Shift A (topology) + Shift B (severity). *Accept:* accuracy + coverage degradation tables per shift, per UQ method, with CIs; a clear statement of which UQ method best flags shift.

**M6 — Rigor + stretch.** Multi-seed + bootstrap CIs across the matrix; ablations; optional Shift C (VMR) qualitative; optional evidential UQ. *Accept:* headline claims carry CIs + ≥3 seeds; ablations support the conclusions.

**M7 — Write-up + release.** `report.pdf` (method, UQ comparison, deferral, shift study, honest limitations, reproduce steps) + `slides.pdf` + README + reproducibility pass; release code (cite Suk CC-BY). *Accept:* a clean clone reproduces the headline figure from `results/`.

---

## 8. Repo layout

```
hemomesh/
  data/            Suk HDF5 → PyG loaders (adapted from repo datasets.py/vtk_demo.py)
  models/          mlp, pointnet, gem_gcn (reuse), mesh_transformer
  uq/              conformal, ensembles, mc_dropout, (evidential)
  metrics/         approximation_error, cosine, calibration (ECE/AUSE), selective_prediction
  shift/           topology + severity-proxy splitters
  train.py  tune.py  plot.py
tests/             metrics + a tiny end-to-end smoke test (pytest)
notebooks/         01_data_eda 02_backbones 03_uq_calibration 04_selective_deferral 05_distribution_shift
results/           experiments.csv (config-hash keyed), figures/
report/  slides/
```

---

## 9. Risks & mitigations

- **Severity proxy is noisy (Shift B).** Mitigate: pick the proxy with the clearest physical meaning (min lumen radius), validate it correlates with WSS/pressure extremes, and report the proxy as a result, not a hidden assumption.
- **Mesh-transformer cost on 24k-node meshes.** Mitigate: multiscale pooling (as repo GEM-GCN does) / patchify; cluster handles it; transformer is the one new backbone, others are cheap/reused.
- **Conformal under heavy shift may be trivially uninformative (coverage collapses).** That is itself the finding (UQ honestly signals "out of distribution"); pair with AUSE so the story isn't binary.
- **Over-scope creep.** Evidential UQ, Shift C, and pulsatile are all explicitly stretch/out — do not let them into the graded core.

---

## 10. Reproducibility

Pinned deps (PyG 2.x, torch, h5py, the repo's GEM-CNN dependency); fixed seeds; `experiments.csv` as the complete run log; dataset MD5s recorded; report carries exact reproduce commands. Mirrors the Biomedical-RAG-Agent reproducibility standard.
