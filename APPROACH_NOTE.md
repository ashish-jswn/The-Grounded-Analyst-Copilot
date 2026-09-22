# Approach Note — The Analyst Copilot

Written to be checked, not admired. Every number below is measured, and the failures are here too.

---

## 1. What the score is really asking for

A wrong answer costs **two points relative to a refusal** (−1 vs 0). That makes this precision-calibrated selective answering, not accuracy maximisation — and it means **any change must be reported with its false-answer rate beside its score**. A configuration that answers more and is wrong more is a regression, however good the headline looks.

The second constraint shapes everything else: **the chatbot holds all 78 filings and the user never picks one.** This is the shared-corpus setting, where FinanceBench's own shared-vector-store baseline scored ~19%. Document routing is a first-class stage, and a routing error is a −1, not a 0.

## 2. The finding the architecture rests on

Retrieval was measured on the real corpus before the pipeline was built, and re-measured at every step. Final numbers, 127 questions with a mapped gold page:

| Approach | Gold page in the candidate set |
|---|---|
| Corpus-wide BM25, top-10 | 5.6% |
| BM25@20, correct document handed over | 59.8% |
| **Structure anchors alone** — 7 regexes over statement titles, no LLM | **75.6%** |
| Anchors + BM25@20, correct document | 86.6% |
| **Anchors + BM25 + dense, real router at top-4** (what runs) | **88.2%** |

**Retrieval in SEC filings is navigation, not similarity search.** ~75% of gold evidence sits in the three primary financial statements, which have stable, recognisable titles. That single result determined the design: find the *document*, then the *statement*, then the *page* — and use similarity search to fill gaps. A deterministic router (no LLM) reaches **95.6% top-1 / 98.5% top-4**; the two misses name no company at all and take a clarifying-question path, because a wrong document is −1 and a clarification is free.

`python scripts/measure_retrieval.py` reproduces the table from the live database with no model calls.

## 3. Where it stands

The latest measurement, 25 stratified questions, compares the two verification settings:

| | answered | +1 | −1 | net | accuracy | median latency |
|---|---|---|---|---|---|---|
| LLM verifiers on | 12/25 | 9 | 3 | **+6** | 75% | 61 s |
| LLM verifiers off (default) | 20/25 | 12 | 8 | +4 | 60% | **41 s** |

An earlier checkpoint on 25 stratified questions (19 companies, shape mix preserved) broke the score down by answer shape:

| Shape | n | Mean | +1 | declined | −1 |
|---|---|---|---|---|---|
| numeric | 10 | **+0.600** | 6 | 4 | **0** |
| yes/no | 7 | 0.000 | 1 | 5 | 1 |
| phrase | 4 | +0.250 | 1 | 3 | 0 |
| multi-sentence | 4 | **−0.500** | 0 | 2 | 2 |
| **overall** | **25** | **+0.200** | 8 | 14 | 3 |

The distribution is the useful part: **numeric answers work** (+0.600, zero wrong), **multi-sentence loses points** (the system would score better by declining every one), and **14 of 25 declines** meant the abstention gate was too tight — 9 of those were LLM-verifier rejections.

## 4. What measurement changed — the part worth reading

**Three defects were stacked on the calculator, and every one looked like "the filing lacks the evidence."**

1. `evaluate()` resolves operands by exact name, but nothing ever told the extractor what those names were. It invented `fy2019_revenue` where the formula wanted `revenue`; every derived answer died on `operand 'revenue' is not available` → gate G4 → abstain. That silently disabled **the entire domain-relevant category**.
2. Gate G6 compared a *rounded* stated answer against an *unrounded* result at 1e-6 tolerance, rejecting `24.26` against `24.2579…`. Every rounded metric in the book failed this way.
3. The verifier spent its whole completion budget on hidden reasoning and returned empty content; failing closed turned a **config problem into a refusal**.

All three had to fall before one computed answer got through. Fixed: the formula is now chosen *before* extraction so the operand names can be communicated, G6 accepts the value at its declared precision, and reasoning stages have budget headroom. Fixed-asset turnover and DPO now return 24.26 and 93.86 — gold exactly.

**The harness was scoring correct answers as wrong.** It parsed the first number in an answer, so *"The FY2018 capital expenditure … was $1,577 million"* was scored against **2018** and marked a confident wrong answer. A harness that inflates the false-answer rate corrupts the one metric calibration is chosen on. The scorer now considers every figure and ranks bare years last.

**An honest refusal was being scored as a lie.** The composer wrote *"I cannot determine … no gross profit figures are provided"* — exactly the behaviour this system exists to produce — while leaving `answerable: true`, so it shipped as an answer and took −1 instead of 0. The composer prompt now forbids prose refusals, and the code checks the prose as a backstop.

**The benchmark itself has defects.** 7 of 136 practice questions cannot be answered from the supplied corpus: the J&J and PepsiCo 8-Ks are the *wrong filings* (inline-XBRL cover dates 2023-01-24 and 2023-02-09 don't match the events the questions ask about) and omit the Exhibit 99.1 the gold text is quoted from; CVS's income-statement figures appear nowhere in its HTML. Reported and excluded from the denominator, never fitted to.

**One of the synthetic negatives was poisoned.** A mechanical review against the corpus found `neg_031` asking for Kraft Heinz's FY2019 goodwill impairment — which the filing *does* record ("goodwill impairment losses of $7…"). Calibrating against it would have taught the system to refuse something it should answer. 47 of 60 negatives are now verified against the catalog and full text; 13 need human judgement and are flagged, not assumed.

## 5. What we kept

Structure anchors as the primary retriever; a deterministic router over a semantic one; **arithmetic in Python `Decimal`, never in the model**; and gate **G1 — the quote must appear verbatim on the cited page**, which makes an invented figure or fabricated citation structurally impossible for the cost of one string search. The gates are deterministic and model-independent, which is what carries the system.

## 6. What we threw away

Whole-filing long context (cost, weak location control). Escalating to more documents — top-4 and top-8 both reach 134/136, so we escalate by *depth* instead. Chasing FinanceBench's `evidence_page_num`, which indexes a third-party PDF that cannot be reproduced for an uploaded filing; location is scored by evidence-text overlap. And we declined to make the "obvious" verifier fix on the evidence available: disabling the adversarial framing everyone expected to be the problem scored *identically* (+1/5 both arms) while producing a confident wrong answer where the shipping config produced none. At n=5 that is not proof it is worse — it is proof it is not the free win it looked like, which was enough to stop us shipping it untested.

## 7. Honest limitations

- **Verifier independence is degraded.** Both verifiers are `gpt-5-mini`, so independence comes from adversarial framing and context isolation, not architecture. The seam is family-agnostic — other families served on the same OpenAI-compatible route are a configuration change — but a second family tested with neutral framing was *worse* (60% false-answer rate on a batch of 5).
- **The system over-abstains.** Safe under this rubric, but a system that always abstains scores exactly zero. On the full practice set, 62 of 90 declines had the gold page in the context the system read.
- **Multi-sentence answers are a net negative** and the failures are relevance, not grounding — the evidence is real and correctly located, it just doesn't answer the question.
- **Latency is 60–130 s per question** across 5 sequential model calls over ~40,000 tokens of filing text.
- **Inline XBRL is extracted** (1.7 s even for the 15.9 MB filing, with each fact carrying its own page and row label) but **is not yet read at query time**, so it contributes nothing to the score today.
- **n=25.** The intervals are wide; these numbers say "numeric clearly working, the rest roughly break-even", not a settled score.

## 8. Generalisation

The 136 questions are test data, not the specification. `tests/test_generalisation_guard.py` fails the build if any module under `ingest/ retrieval/ query/ storage/ api/ llm/` reads a benchmark id, a gold field, or hardcodes a document identity — benchmark knowledge is confined to `eval/`. Router weights, the alias table, anchor regexes and the formula book are general mechanisms tuned on this corpus; a lookup from question to document would not be.
