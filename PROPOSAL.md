# Project 2 Proposal (v2) — HemoMesh: Calibrated Uncertainty & Selective Deferral for Coronary Wall-Shear-Stress Surrogates under Distribution Shift

**Course:** MTH 5320 (Deep Learning), Summer 2026
**Constraint:** must use an architecture more advanced than a dense network. **Satisfied:** mesh/message-passing GNN, SE(3)-equivariant mesh GNN, and/or mesh transformer — with a dense MLP/PointNet retained explicitly as the baseline to beat.
**Working name:** `HemoMesh`.
**Publication target (realistic):** MICCAI-STACOM / GeoMedIA / ML4H workshop, or a short methods note.

> **v2 note.** This proposal was rewritten after an adversarial pressure-test, independent methodological review, and a dataset-availability probe. The original v1 (pressure/vFFR target + self-generated SimVascular CFD over 1.5–2k cases + 4×3 backbone/UQ sweep) is archived at `PROPOSAL-v1-superseded.md`. See §10 for exactly what changed and why.

---

## 1. Problem framing

CFD is the gold standard for coronary hemodynamics — pressure, velocity, and **wall shear stress (WSS)**, the mechanical signal most implicated in atherosclerotic plaque progression. CFD is slow (hours/case). A **learned surrogate** maps vessel geometry → hemodynamic fields in milliseconds. The 2024 Suk et al. and 2025 Nannini benchmark work already showed geometric-DL surrogates can predict these fields accurately, and that transformer/equivariant backbones generalize best.

What those works **did not** do — and what makes a surrogate trustworthy enough to actually replace CFD — is **uncertainty**. A point-estimate WSS field with no error bars cannot be deployed: you cannot tell where it is wrong, and CFD is hardest exactly where it matters (stenoses, bifurcations). This project's contribution is **calibrated, spatially-resolved uncertainty for coronary WSS surrogates, with a selective-deferral protocol, evaluated under controlled distribution shift.**

### Thesis (the contribution, and the eval-rigor brand)
> A coronary hemodynamic surrogate is only usable if it knows where it is wrong. We add calibrated per-node uncertainty to mesh-based WSS surrogates and show it (a) holds nominal coverage in-distribution, (b) **degrades measurably under distribution shift** (and which UQ method degrades most informatively), and (c) enables **selective deferral** — flagging the highest-uncertainty regions/cases to fall back to full CFD, so error on the retained predictions drops along a risk–coverage curve. The headline is the gap between how accurate these surrogates *look* and how *trustworthy* they are once the geometry distribution moves.

This is the research-track sibling of Project 1: the "score per item" becomes a "WSS vector per mesh node," the architecture jumps from dense MLP to geometric/attention, and Project 1's rigor spine (held-out protocol, multi-seed, bootstrap CIs, honest failure analysis) carries straight over.

---

## 2. Data — reuse, no CFD generation

**Primary dataset: Suk et al., `coronary-mesh-convolution`** — https://github.com/sukjulian/coronary-mesh-convolution (MIT code; **dataset CC-BY 4.0**). **Download verified (2.5 GB zip pulled and inspected, 2026-06-30).** Synthetic coronary arteries generated in SimVascular (incompressible Navier–Stokes), steady-flow: **3,999 cases total — 2,000 `single` + 1,999 `bifurcating`.** Each case (HDF5 group `sample_NNNN`) stores **per-node `wss` (N×3 vector) AND per-node `pressure` (N×1 scalar)**, plus `pos` (N×3), `face` (triangles), `inlet_idcs`. Units CGS, WSS/pressure in dyn/cm² (=0.1 Pa). Mesh sizes ~5.9k–11.8k nodes (single, mean 9.5k) and ~9.8k–24.3k (bifurcating, mean 16.8k). PyTorch-Geometric `raw`/`processed` convention. **Pretrained baselines included** (GEM-GCN, DiffusionNet) as a harness sanity-check.

**This removes the single highest-risk assumption of v1** (building and validating a SimVascular CFD pipeline). The project trains on existing labels.

**Targets: per-node WSS (vector) and pressure (scalar)** — both verified present. **vFFR / clinical diagnostic claims remain out of scope:** these are synthetic geometries with no invasive ground truth, so any derived pressure-ratio is reported only as "agreement with the CFD pseudo-label," never as clinical diagnostic accuracy. (The only dataset with curated vFFR labels, Nannini 2501.09046, is "available on request" and unverifiable — not relied upon.) **Pulsatile data is NOT in this bundle** (steady-flow only), so the steady→pulsatile shift is excluded unless separately sourced.

**Optional real-geometry stress test (stretch):** a handful of real coronary surface meshes from the Vascular Model Repository (geometry only) for a *qualitative* check — does predicted uncertainty spike on real anatomy? Reported as exploratory, no clinical metrics.

---

## 3. Comparison axes

### Axis A — Backbone (secondary; satisfies the architecture requirement)
A small, deliberate set — **not** the contribution, just enough to show the UQ findings are backbone-robust and to beat the dense baseline:
| Model | Role |
|---|---|
| **MLP / PointNet** (per-node, no message passing) | the dense-network baseline the course requires beating; the "is geometry helping?" control |
| **Mesh / message-passing GNN** (MeshGraphNet-style) | standard geometric backbone |
| **SE(3)-equivariant mesh GNN *or* mesh transformer** | one strong backbone (Suk's equivariant net is reusable; the mesh-transformer is the benchmark's real-data winner) |

### Axis B — Uncertainty quantification (the contribution)
| Method | Type | Role |
|---|---|---|
| **Split / conformal prediction** (conformalized per-node residuals) | distribution-free, calibrated coverage | **primary** — coverage is exact in-distribution and breaks measurably under shift |
| **Deep ensembles** | epistemic | the strong learned-UQ reference |
| **MC dropout** | cheap epistemic | low-cost baseline |
| *Evidential / heteroscedastic regression* | single-pass | **stretch**, only if the above land |

UQ being multi-method is justified here because it *is* the spine, not an extra axis.

---

## 4. Evaluation plan

### WSS field accuracy
- NMAE / RMSE per node; **cosine similarity** for WSS vector direction; approximated-WSS error (as in Suk et al.) for comparability.

### Uncertainty quality (the core)
- **Coverage vs. nominal** (conformal): does the 90% interval contain truth 90% of the time, in-distribution and under shift?
- **Regression ECE** + reliability diagrams; **error–uncertainty correlation** / **AUSE** (area under sparsification error).

### Selective prediction / deferral
- **Risk–coverage curves:** defer the top-k% highest-uncertainty nodes/cases to "full CFD"; show retained-set error drops. Quantify deferral efficiency.
- **Spatial analysis:** does uncertainty concentrate at stenoses/bifurcations (the hard regions)?

### Distribution shift (the headline)
- Controlled, fully in-data shifts: **single → bifurcating** (topology), **mild → severe stenosis** (severity), **steady → pulsatile** (regime). Measure coverage degradation and which UQ method's uncertainty best signals the shift.

### Statistical rigor (the brand)
- **Bootstrap CIs** on every headline metric; **multi-seed** training; ablations (backbone, UQ method, conformal calibration-set size).

---

## 5. Deliverables (mirrors Project 1)

- **`hemomesh/` package** — dataset loaders (Suk HDF5/PyG), mesh featurization, models (MLP/PointNet, mesh GNN, equivariant/transformer), UQ wrappers (conformal, ensembles, MC-dropout), metrics (WSS + calibration + selective-prediction), plotting. Unit tests for metrics + a smoke test (pytest), reusing Project 1's harness shape.
- **Resumable tuning campaign** → `results/experiments.csv`, keyed by config hash (lift Project 1's logging wholesale).
- **Notebooks:** `01_data_eda`, `02_backbones`, `03_uq_calibration`, `04_selective_deferral`, `05_distribution_shift`.
- **`report.pdf`** (paper-style: method, UQ comparison, selective-deferral, shift study, honest limitations, reproduce steps) + **`slides.pdf`**.

---

## 6. Milestones (Project-1 cadence) — note the de-risked M1

1. **Data + baseline reproduction.** Download Suk dataset; confirm case count; **reproduce the published WSS baseline** to validate the harness end-to-end. *(No CFD generation — this is the de-risked replacement for v1's CFD-pipeline milestone.)*
2. **Backbone set (Axis A).** MLP/PointNet baseline → mesh GNN → one strong backbone; in-distribution WSS accuracy; staged tuning w/ config-hash logging.
3. **UQ methods (Axis B).** Conformal (primary) + deep ensembles + MC-dropout; calibration + error-correlation eval in-distribution.
4. **Selective deferral.** Risk–coverage curves; uncertainty-localizes-at-stenoses analysis.
5. **Distribution-shift study (headline).** Define shifts (single→bifurcation, mild→severe, steady→pulsatile); coverage degradation + which UQ flags shift best.
6. **Rigor + stretch.** Multi-seed + bootstrap CIs + ablations; optional qualitative real-geometry (VMR) stress test.
7. **Write up.** `report.pdf` + `slides.pdf` + README + reproducibility; release code.

---

## 7. Why this scores on the research/grad-school criteria

- **Novel + builds on existing research:** stands on Suk's equivariant WSS surrogate and the 2025 benchmark, and attacks the gap they *both* leave open — there is **no calibrated UQ or selective-deferral protocol** for coronary hemodynamic surrogates. That, not the backbone comparison, is the delta.
- **Publishable and de-risked:** the contribution (UQ + shift + deferral) is feasible on a verified-downloadable dataset with pretrained baselines; it publishes even if no backbone "wins," because the finding is about *trustworthiness under shift*.
- **Honest:** WSS field regression + calibration, no clinical overclaim, no synthetic-FFR threshold theater.
- **Fills the portfolio gap + threads health-ML:** adds the graph-network/geometric-DL class, in cardiology, extending the SkinCheck / brain-tumor / biomedical-RAG story.
- **Real engineering:** geometric-DL training + a multi-method UQ harness + selective-prediction + a controlled-shift protocol is substantial, and it's squarely the evaluation-rigor lane.

## 8. Key prior work (anchors)
- Suk, de Haan, Lippe, Brune, Wolterink, *Mesh Neural Networks for SE(3)-Equivariant Hemodynamics Estimation on the Artery Wall* (Comput. Biol. Med. 2024; arXiv:2212.05023) — **the dataset + WSS-surrogate baseline.**
- Nannini, Suk et al., *Learning Hemodynamic Scalar Fields on Coronary Artery Meshes: A Benchmark of Geometric DL Models* (arXiv:2501.09046, 2025) — backbone-generalization context.
- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks* (MeshGraphNets, ICLR 2021) — lineage.
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (2021) — UQ method.
- Lakshminarayanan et al., *Deep Ensembles* (NeurIPS 2017); Gal & Ghahramani, *MC Dropout* (ICML 2016).

## 9. Open decisions to lock before scaffolding
1. **Strong backbone:** SE(3)-equivariant (reuse Suk's, strongest accuracy) vs. mesh-transformer (benchmark's generalizer) — or both if time? *(Recommend transformer: differentiates from Suk and lets you test the "transformer generalizes" claim through the UQ lens.)*
2. **UQ core vs. stretch:** conformal + ensembles core, MC-dropout baseline, evidential stretch — confirm.
3. **Shift axes:** all three, or focus two (single→bifurcation + mild→severe) as core with pulsatile as stretch? *(Recommend two core + pulsatile stretch.)*
4. **Real-geometry stress test:** include as qualitative stretch, or omit for focus?
5. **Repo:** standalone `HemoMesh` *(recommend yes)*.

## 10. What changed from v1, and why (pressure-test + data probe)
- **Target pressure/vFFR → WSS.** v1 depended on pressure/vFFR labels that have no verifiable public dataset; WSS has a downloadable, labeled, coronary-specific one. Also removes the not-well-posed "diagnostic accuracy at 0.80" claim.
- **Self-generated CFD → reuse Suk's dataset.** Eliminates v1's highest-risk assumption (a solo SimVascular pipeline producing 1.5–2k QA'd, BC-correct coronary CFD cases on the timeline — judged the project-killer).
- **Backbone sweep demoted, UQ promoted.** v1's 4-backbone comparison duplicated prior work; the genuine novelty is the calibrated-UQ + selective-deferral + shift protocol, now the spine. Backbones cut to ~3 (incl. the dense baseline).
- **Real "test benchmark" → exploratory stress test.** VMR has only a few dozen coronary cases; not a quantitative benchmark. Shift is now measured on controlled *in-data* splits instead.
- **UQ methods 3→ a focused conformal-led set;** metrics renamed to honest WSS/calibration language; pulsatile + mesh-resolution ablations moved to stretch.
