# Project 2 Proposal — Neural Reranking for Biomedical Evidence Retrieval

**Course:** MTH 5320 (Deep Learning), Summer 2026
**Constraint:** must use an architecture more advanced than a dense network. **Satisfied:** transformer cross-encoder + late-interaction (ColBERT-style) models.
**Working name:** `EvidenceRank` — a neural reranking stage for the [Biomedical-RAG-Agent](https://github.com/Lawson-Darrow/Biomedical-RAG-Agent).

---

## 1. Problem framing

A retrieval-augmented generation (RAG) system answers a question by (1) retrieving candidate passages, then (2) synthesizing a grounded, cited answer. The existing Biomedical-RAG-Agent does first-stage retrieval well — BM25 + dense (BGE-small, ONNX) fused with Reciprocal Rank Fusion over pgvector — but it **stops there**: the order the synthesizer sees is RRF's order. First-stage retrievers score query and passage *independently* (a dot product of two embeddings), so they cannot model fine-grained term interaction ("does *this* passage actually support *this* claim?").

A **reranker** is a second stage that re-scores the top-*k* candidates with a model that sees the query and passage **jointly**. This is the standard two-stage IR architecture behind production search and RAG. The project builds, compares, and rigorously evaluates neural rerankers, then integrates the winner into the live agent and measures the **downstream** effect on answer faithfulness — not just retrieval metrics.

This is the direct sequel to Project 1: Project 1 was *feature-based* learning-to-rank with an MLP (pointwise regression / pointwise classification / pairwise RankNet). Project 2 is *text-based* learning-to-rank with a transformer — **same loss families, more powerful encoder, real downstream task.**

### Thesis (the through-line)
> A learned reranker's value is conditional on the evaluation setup. On a small/easy corpus it is invisible; on a deep, hard candidate list it is decisive. Measure it honestly across both, and report the latency cost it buys with.

---

## 2. Two comparison axes (the spine of the study)

The rigor structure mirrors Project 1's "compare N formulations on a shared backbone." Here there are **two crossed axes**, which is what gives the project its depth:

### Axis A — Architecture (the "advanced architecture" requirement)
| Model | Interaction | Cost | Role |
|---|---|---|---|
| **Bi-encoder** (BGE-small, existing) | none (independent embeddings) | cheap, indexable | **baseline** — the system as it exists today |
| **Cross-encoder** | full token×token attention over [query; passage] | expensive (no pre-indexing) | accuracy ceiling |
| **ColBERT-style late interaction** | token-level MaxSim | mid (partially indexable) | the speed/quality middle ground |
| **Distilled student** (stretch) | cross-encoder → bi/late-interaction via MarginMSE/KL | cheap | recover accuracy at low latency |

### Axis B — Loss formulation (the Project-1 callback)
The same three ranking-loss families from Project 1, now on a transformer:
- **Pointwise** — binary cross-entropy on (query, passage) relevance.
- **Pairwise** — RankNet / margin loss on (relevant, non-relevant) pairs *(direct Project-1 continuity)*.
- **Listwise** — softmax cross-entropy / LambdaLoss over the candidate list, optimizing the ranking directly.

Crossing A×B (with B focused on the cross-encoder, where it matters most) gives a clean, gradeable matrix and a real research question: *which loss formulation best trains a transformer reranker for biomedical IR, and how much does cross-document/listwise context help over pointwise?*

---

## 3. Data

### Training (general domain → transfer)
- **MS MARCO passage ranking** — the standard large-scale reranker training set (qrels + mined hard negatives). Train the general reranker here.

### IR evaluation (hard, judged biomedical benchmarks)
Standard zero-shot biomedical IR sets with relevance judgments (qrels), so nDCG / MRR / Recall are properly defined and comparable to the literature:
- **BEIR: TREC-COVID, NFCorpus, SciFact, BioASQ.**
- First stage returns top-100 (BM25 + dense, reproducing the agent's RRF); the reranker reorders that list.

### End-to-end RAG evaluation (the "useful tool" payoff)
- The agent's **own corpus** (PubMedQA contexts + PMC-OA slice) and **existing eval harness** — faithfulness, hallucination rate, citation accuracy, task accuracy (PubMedQA), abstention correctness. Measured **closed-book vs. RRF-only vs. RRF+reranker** across the same frontier-vs-open model panel already wired through the production inference gateway.

### Domain adaptation (depth)
Train general (MS MARCO) → evaluate zero-shot on biomedical → then **domain-adapt** and measure the lift:
- Swap the cross-encoder backbone to a biomedical encoder (**PubMedBERT** / **BioLinkBERT**) vs. a general one (MiniLM) — does in-domain pretraining help the reranker?
- Optional: **GenQ-style pseudo-labeling** (generate synthetic queries over the biomedical corpus) for unsupervised in-domain fine-tuning.

All backbones are OSS (honors the prefer-OSS-tools principle); all training is designed for a Linux GPU/runtime environment, independent of the production inference gateway (which is inference-only and unavailable for embeddings).

---

## 4. Evaluation plan

### Retrieval quality (on BEIR biomedical)
- **nDCG@10** (primary), **MRR@10**, **Recall@{10,100}**, **MAP**.
- Reranker reorders first-stage top-100; report metric deltas vs. the RRF baseline.

### End-to-end RAG quality (on the agent's corpus)
- Reuse the existing judge-based suite: **faithfulness, hallucination rate, citation accuracy, task accuracy, abstention correctness** — closed vs. RRF vs. RRF+reranker, across the model panel.

### Efficiency (the production tradeoff)
- **Per-query rerank latency, throughput, memory/index footprint** → plot the **latency↔nDCG Pareto frontier** across the architecture axis. This is the engineering core: cross-encoder wins quality, ColBERT/distilled wins the frontier.

### Statistical rigor (the brand)
- **Bootstrap confidence intervals** on every headline metric; **paired significance tests** (per-query) between systems; **multi-seed** training reruns for the final comparison.
- **The honest-eval result:** run the *same* reranker on (a) the original small/easy corpus and (b) the deep BEIR setting, and show the gain is ~0 in (a) and significant in (b) — measurement design determines the conclusion.

---

## 5. Deliverables (mirrors Project 1)

- **`evidencerank/` package** — first-stage candidate generation, reranker models (cross-encoder, ColBERT, distillation), training loop, loss families, metrics, plotting. Unit tests for metrics + models (pytest), reusing Project 1's harness shape.
- **Resumable tuning campaign** logged to `results/experiments.csv`, keyed by config hash (lift Project 1's logging wholesale).
- **Notebooks:** `01_eda` (corpus/candidate analysis), `02_tuning`, `03_results`, `04_rigor` (CIs, baselines, small-vs-deep corpus study, latency).
- **Integration patch** into Biomedical-RAG-Agent: a rerank stage between RRF fusion and synthesis, behind a config flag, with the end-to-end before/after table.
- **`report.pdf`** (paper-style: method, A×B results, domain-adaptation, latency Pareto, honest limitations, reproduce steps) + **`slides.pdf`** (10-min deck).

---

## 6. Milestones (Project-1 cadence)

1. **Spec + scaffold + data.** Package layout (reuse Project 1 harness); acquire MS MARCO triples + BEIR biomedical sets (qrels). Reproduce first-stage BM25+dense top-100 + RRF.
2. **Reranking eval harness.** nDCG@10/MRR/Recall + bootstrap CIs over reranked top-100; verify on the RRF baseline (no learning yet).
3. **Cross-encoder, pointwise.** Train on MS MARCO; zero-shot eval on BEIR biomedical. First real reranking lift.
4. **Loss-formulation sweep (Axis B).** Pointwise vs pairwise (RankNet) vs listwise on the cross-encoder; staged tuning campaign w/ config-hash logging; pick the winning formulation.
5. **Architecture sweep (Axis A).** ColBERT late-interaction + distillation; build the latency↔nDCG Pareto frontier.
6. **Domain adaptation + rigor study.** PubMedBERT/BioLinkBERT backbone vs general; (optional) GenQ pseudo-labels; the small-vs-deep-corpus honest-eval result; multi-seed final reruns + CIs.
7. **Integrate + write up.** Wire the winner into Biomedical-RAG-Agent; end-to-end faithfulness/hallucination/citation/task lift table; `report.pdf` + `slides.pdf` + README + reproducibility pass.

---

## 7. Why this scores on all three of Lawson's criteria

- **Job signal:** "production two-stage RAG with a trained, distilled reranker served under a latency budget" is precisely the applied-LLM/search-engineering ask; it ties to and visibly upgrades an existing flagship repo + live demo.
- **Useful tool:** the winner ships into the real agent — a measurable answer-faithfulness improvement, not a throwaway notebook.
- **Depth / real engineering:** two crossed comparison axes, knowledge distillation, domain adaptation, a serving-latency Pareto study, and statistically honest evaluation — well beyond a single fine-tune.

## 8. Open decisions to lock before scaffolding
1. **Core vs stretch scope:** is distillation (Axis A, row 4) in the graded core, or a stretch goal? (Recommend: stretch — the A×B core is already a full project.)
2. **Backbone for the cross-encoder:** start from a general MiniLM and add PubMedBERT as the domain-adaptation arm (recommended), or go biomedical-first?
3. **BEIR set count:** all four (TREC-COVID/NFCorpus/SciFact/BioASQ) or a focused two for depth-over-breadth?
4. **Repo:** new standalone `EvidenceRank` repo that depends on the agent, or a feature branch inside Biomedical-RAG-Agent? (Recommend: standalone repo for a clean portfolio artifact, integrated via PR.)
