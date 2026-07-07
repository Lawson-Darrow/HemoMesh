# Project 2 Proposal — HemoMesh: A Calibrated, Generalizable Mesh-Transformer Surrogate for Coronary Hemodynamics & Virtual FFR

**Course:** MTH 5320 (Deep Learning), Summer 2026
**Constraint:** must use an architecture more advanced than a dense network. **Satisfied:** message-passing GNN, SE(3)-equivariant mesh GNN, and mesh/point transformer — with a dense MLP/PointNet retained explicitly as the baseline to beat.
**Working name:** `HemoMesh`.
**Publication target (realistic):** MICCAI-STACOM / ML4H / GeoMedIA workshop, or a short journal note (*Computers in Biology & Medicine*, *Medical Image Analysis*).

---

## 1. Problem framing

Coronary CFD (computational fluid dynamics) is the gold standard for computing patient-specific hemodynamics — pressure, velocity, and wall shear stress (WSS) — and for deriving **Fractional Flow Reserve (FFR)**, the clinical decision metric for whether a stenosis is functionally significant (FFR ≤ 0.80 → revascularize). Invasive FFR needs a pressure wire in the cath lab; **FFR-CT** (the HeartFlow product) computes it non-invasively from CTA via CFD, but each case takes hours of offline solving. A **learned surrogate** maps vessel geometry → hemodynamic fields in milliseconds, enabling real-time, on-site FFR.

Two things block trust in such surrogates today, and they define this project:

1. **Generalization, synthetic→real.** Labels come from CFD, and dense patient cohorts with CFD ground truth barely exist (~hundreds), so the field trains on large *synthetic* geometry cohorts. The 2025 geometric-DL hemodynamics benchmark (Nannini/Suk et al.) showed most backbones look great on synthetic test data but **only transformer-based backbones generalize to real patient-specific geometries** — and even then only qualitatively. Robust, *quantified* synthetic→real transfer is unsolved.
2. **Uncertainty.** A black-box FFR with no error bars is clinically unusable. Current surrogates give a point estimate; **calibrated per-node spatial uncertainty** — "the surrogate is unsure *right here, near this stenosis*" — is essentially absent.

### Thesis (the through-line — and the eval-rigor brand)
> A coronary hemodynamic surrogate is only meaningful if it (a) generalizes from synthetic training geometries to real patient anatomy and (b) tells you where it is untrustworthy. We build an equivariant mesh-transformer surrogate for per-node pressure/WSS → vFFR with calibrated spatial uncertainty, and evaluate it under a rigorous synthetic→real protocol the field currently lacks. The headline result is the gap between how good these models *look* on synthetic data and how they *behave* on real geometries — and a UQ signal that flags exactly where they fail.

This is the research-track sibling of Project 1: Project 1 was learning-to-rank with an MLP; here the "score per item" becomes a "field per mesh node," the architecture jumps from dense to geometric/attention, and the rigor spine (held-out protocol, multi-seed, CIs, honest failure analysis) carries straight over.

---

## 2. Two comparison axes (the spine of the study)

Mirrors Project 1's "compare N formulations on a shared problem." Two crossed axes:

### Axis A — Backbone (the architecture requirement + the generalization question)
| Model | Geometry coupling | Role |
|---|---|---|
| **MLP / PointNet** (per-node features, no message passing) | none | **baseline** — the dense-network the course wants beaten; also the "is geometry even helping?" control |
| **Message-passing GNN** (GraphSAGE / MeshGraphNet-style) | local neighborhood | standard mesh GNN |
| **SE(3)/gauge-equivariant mesh GNN** (Suk et al. style) | local + rotation-equivariant | tests whether equivariance buys real-data robustness |
| **Mesh/point transformer** (global attention) | global | the benchmark's real-data winner |

**Central question:** does the equivariant/transformer backbone actually close the synthetic→real gap vs. vanilla GNN/MLP — and by how much, with confidence intervals? That is a clean, gradeable, publishable comparison.

### Axis B — Uncertainty quantification
| Method | Cost | Role |
|---|---|---|
| **Deep ensembles** | N× training | gold-standard reference |
| **MC dropout** | 1× train, N× inference | cheap baseline |
| **Evidential / heteroscedastic regression** | single pass | the efficient contribution |

Evaluate which gives **calibrated per-node** uncertainty against a CFD ensemble and against actual error.

**Targets:** per-node **pressure**, per-node **WSS vector field**, and the derived scalar **vFFR**.

---

## 3. Data

### Geometries
- **Vascular Model Repository (VMR, vascularmodel.org)** — ~200 patient/animal-specific vascular models (coronary subset) as curated SimVascular projects. Real geometries; free. → the **real-geometry test set**.
- **Synthetic coronary cohort (generated)** — parametric / statistical-shape-model generator producing **1,500–2,000 coronary trees** (single + bifurcating, varying stenosis severity/location). → the **training set**. This is the proven paradigm (the benchmark used 1,500 synthetic LCA bifurcations; Suk et al. ~5,000 CFD sims).

### Labels (generated via CFD)
- **SimVascular / svSolver** for 3D steady-state CFD → dense pressure/WSS fields (~minutes/case, embarrassingly parallel on the university GPU cluster).
- **svZeroDSolver** (0D/1D reduced-order) as a cheap label-bootstrap and a physics baseline.
- **Pipeline validation first:** reproduce VMR-provided reference CFD results on a handful of cases before scaling — de-risks the single biggest correctness threat.

### Split (the headline protocol)
- **Train/val:** synthetic cohort.
- **Test:** held-out **real patient-specific** VMR coronaries (+ any patient-specific CFD cases), scaled-down analog of the benchmark's 427-case real test. The **synthetic-test vs. real-test gap, per backbone, is the central result.**

All tools (SimVascular, PyTorch Geometric) are OSS — honors the prefer-OSS-tools principle. Compute is over-provisioned, which is the advantage: the cluster's real value is parallel CFD label generation + large ablation sweeps, letting the project win on rigor.

---

## 4. Evaluation plan

### Field accuracy
- **Pressure / WSS:** MAE, RMSE, R², normalized MAE (NMAE, as in Suk et al.); **cosine similarity** for WSS vector direction.

### Clinical metric (vFFR)
- vFFR correlation vs. CFD; mean error; **% within ±0.05**; **diagnostic accuracy / sensitivity / specificity at the 0.80 threshold** (the actual clinical decision); Bland–Altman agreement.

### Generalization (the central finding)
- Synthetic-test vs. real-test metric gap, **per backbone** — quantifying the documented "synthetic looks great, real collapses" failure and which architecture survives it.

### Uncertainty quality
- Reliability diagrams + **regression ECE**; error-vs-uncertainty correlation; spatial analysis: **does predicted uncertainty spike near stenoses/bifurcations** (where CFD itself is hardest)? Decision-useful UQ = flagging the cases/regions to defer to full CFD.

### Efficiency
- Inference latency vs. CFD wall-clock — the orders-of-magnitude speedup story.

### Statistical rigor (the brand)
- **Bootstrap CIs** on every headline metric; **multi-seed** training; ablations (equivariance on/off, UQ method, **mesh-resolution sensitivity**, synthetic-cohort size).
- **Honest-eval headline figure:** a backbone with synthetic R²≈0.99 whose vFFR diagnostic accuracy drops on real geometries — and the calibrated UQ that flags exactly those failures.

---

## 5. Deliverables (mirrors Project 1)

- **`hemomesh/` package** — geometry loading, SimVascular CFD-label pipeline wrappers, mesh featurization (geodesic-to-inlet, curvature, normals), models (MLP/PointNet, GNN, equivariant GNN, mesh-transformer), UQ heads, training loop, metrics, plotting. Unit tests for metrics + a tiny end-to-end smoke test (pytest), reusing Project 1's harness shape.
- **Resumable tuning campaign** → `results/experiments.csv`, keyed by config hash (lift Project 1's logging wholesale).
- **Notebooks:** `01_geometry_eda`, `02_cfd_label_validation`, `03_tuning`, `04_results`, `05_uq_and_generalization`.
- **`report.pdf`** (paper-style: method, both axes, generalization protocol, UQ, honest limitations, reproduce steps) + **`slides.pdf`** (10-min deck).

---

## 6. Milestones (Project-1 cadence)

1. **Scaffold + CFD pipeline.** Package layout (reuse Project 1 harness); pull VMR coronary geometries; stand up SimVascular; **validate svSolver output against VMR reference** on a few cases.
2. **Data generation.** Parametric/SSM coronary generator → 1,500–2,000 geometries → batch CFD labels (pressure/WSS) on the cluster; build the mesh dataset (PyG).
3. **Baselines + harness.** MLP/PointNet baseline; full metric suite (field + vFFR + UQ) with bootstrap CIs; verify on synthetic test (no real-data peeking yet).
4. **Architecture sweep (Axis A).** GNN → equivariant GNN → mesh-transformer; synthetic-test leaderboard; staged tuning w/ config-hash logging.
5. **Generalization study.** Unlock the real patient-specific test set; report the synthetic→real gap per backbone — the central result.
6. **UQ (Axis B) + rigor.** Deep ensembles / MC-dropout / evidential; calibration eval; uncertainty-localizes-near-stenoses analysis; multi-seed + ablations.
7. **Integrate + write up.** `report.pdf` + `slides.pdf` + README + reproducibility pass; package the CFD-label + training pipeline for release.

---

## 7. Why this scores on the research/grad-school criteria

- **Novel + builds on existing research:** stands directly on the 2025 geometric-DL hemodynamics benchmark, Suk's SE(3)-equivariant mesh nets, and Pegolotti's GNN-ROMs — and attacks their *named* open problems (real-geometry generalization, calibrated UQ) rather than re-deriving a crowded base result.
- **Publishable:** even if synthetic→real generalization only partly closes, the **calibrated-UQ contribution and the standardized real-generalization protocol are publishable in their own right** (workshop-grade), so the project de-risks into a result regardless.
- **Actually useful:** non-invasive, real-time FFR is a live clinical/commercial target (FFR-CT); a *trustworthy* surrogate that knows when to defer to full CFD is a genuine contribution.
- **Fills the portfolio gap + threads the health-ML narrative:** adds the graph-network/geometric-DL class you're missing, in cardiology — extending the SkinCheck / brain-tumor / biomedical-RAG "serious health ML across modalities" story.
- **Real engineering:** a CFD label-generation pipeline + geometric-DL training + UQ + a serving-latency story is substantial systems work, not a single fine-tune.

## 8. Key prior work (anchors)
- Nannini, Suk, Rygiel, Saitta et al., *Learning Hemodynamic Scalar Fields on Coronary Artery Meshes: A Benchmark of Geometric DL Models* (arXiv:2501.09046, 2025) — the gap-definer.
- Suk, de Haan, Lippe, Brune, Wolterink, *Mesh Neural Networks for SE(3)-Equivariant Hemodynamics Estimation on the Artery Wall* (Comput. Biol. Med. 2024; arXiv:2212.05023).
- Pegolotti et al., *Learning Reduced-Order Models for Cardiovascular Simulations with GNNs* (Comput. Biol. Med. 2023).
- Rygiel, Suk et al., *Active Learning for DL-Based Hemodynamic Parameter Estimation* (arXiv:2503.03453, 2025) — data-efficiency angle.
- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks* (MeshGraphNets, ICLR 2021) — lineage.
- Vascular Model Repository (vascularmodel.org); SimVascular (simvascular.github.io).

## 9. Open decisions to lock before scaffolding
1. **CFD fidelity:** steady-state core (recommended) with pulsatile as a stretch? Steady-state is minutes/case and sufficient for the generalization+UQ story; pulsatile multiplies cost.
2. **Output domain:** 3D surface-mesh wall fields (recommended — that's where geometric-DL value + the equivariance story live) vs. centerline/1D ROM (use only as a cheap label-bootstrap/baseline).
3. **Vessel bed:** coronary-only (recommended — FFR is coronary) vs. include aorta.
4. **UQ scope:** deep ensembles (primary) + evidential (single-pass contribution), with MC-dropout as the cheap baseline — or all three fully?
5. **Synthetic generator:** build a parametric coronary generator vs. statistical shape model from VMR — affects M2 effort.
6. **Repo:** standalone `HemoMesh` repo (recommended, clean portfolio artifact).
