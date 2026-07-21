# ACEC v5 Marginal Evidence Utility Standard

## 1. Normative target

For question `q`, evidence available before retrieval turn `t`, `E_<t`, and
the exact document selected by the runtime verifier, `d_t`, ACEC v5 supervises

```text
DeltaU_t = U(q, E_<t union {d_t}) - U(q, E_<t).
```

`DeltaU_t` is the formal target. A document's title, dataset id, or membership
in a retrieval batch is never a target. Static relevance or support is only an
intermediate observation: a supporting document can have zero marginal utility
when its information was already covered.

The comparison is conditional on history and must be computed in trajectory
order. Evidence retrieved in the same turn is compared with the same `E_<t`;
one document may legitimately cover several distinct requirements.

## 2. Valid utility providers

The standard permits multiple providers, but an artifact must declare which
provider produced every row.

1. **Requirement coverage.** When a dataset supplies auditable evidence spans,
   `U` is the weighted fraction of minimal evidence requirements satisfied.
   This is the current HotpotQA v5 builder's provider.
2. **Answer score.** With a gold answer, `U` may be normalized EM, F1, answer
   log-probability, or another frozen outcome score evaluated before and after
   adding the document.
3. **Frozen judge.** Without evidence annotations, a versioned evaluator may
   estimate answerability before and after the document is added.
4. **Human audit.** A blinded human comparison may directly assess the delta.

Scores from different providers must not be pooled as interchangeable numbers
unless provider identity is modeled and cross-provider calibration is reported.

## 3. Requirement representation

A requirement is one minimal semantic piece needed to answer the question. It
contains one or more evidence units with exact text spans. Source ids, titles,
sentence ids, citations, and dataset fields are provenance only.

For HotpotQA, supporting sentences are grouped by source paragraph into a
requirement. For another dataset, an adapter may use decomposition steps,
citations, database rows, or human rationales. New adapters should emit the
canonical `EvidenceSpecification` contract instead of changing the calibrator.

`K` is the number of minimal evidence requirements, not a hard-coded number of
hops or titles. Auto-K may select a learned predictor only when it improves on
a modal fixed-K baseline on question-disjoint validation data.

## 4. Assessability and abstention

Every row records:

- utility before and after adding the document;
- marginal utility;
- utility provider and confidence;
- exact selected document and assigned requirement;
- static support assessment and its evidence span;
- document novelty relative to prior evidence;
- whether the row is eligible for fitting.

Ambiguity is not converted into a negative example. A row is excluded when its
slot-to-requirement assignment is weak or ambiguous, its evidence span is
missing, or the annotation scope cannot justify absence. Closed candidate-pool
annotations may justify a zero-support assessment; support-only annotations
may not.

## 5. Current Bernoulli interface

The v5 audit retains continuous `DeltaU`. The current ACEC filter consumes a
Bernoulli event: whether the selected document covers a previously unsatisfied
requirement. Only high-confidence endpoint assessments enter that fit.

The observation model uses two independent runtime features:

- NLI support between the selected document and slot hypothesis;
- document novelty relative to evidence selected in prior turns.

Both coefficients are constrained non-negative. Repeated documents therefore
cannot be treated like independent new evidence merely because they retain a
high NLI score.

## 6. Splits and gates

Splits are made by complete question/trajectory, never by individual step. The
default 500-record calibration pilot uses 300 fit, 100 validation, and 100 test
records so the operating point has enough tail examples.

An artifact is written only when the validation gates pass:

- sufficient assessable examples;
- posterior ROC-AUC and average precision minima;
- a minimum assessable-row fraction, so abstention cannot leave only an easy
  endpoint subset;
- a novelty-only baseline and a non-repeat semantic-support audit, so exact
  duplicate detection cannot masquerade as evidence understanding;
- minimum target examples for every action mode claimed by the artifact;
- stable dataset question ids and corpus/document ids at configured coverage;
- a binding threshold chosen on validation that reaches the required precision
  with a minimum number of predicted gains.

The selected threshold is frozen before test evaluation. Test metrics are
reported once and are not used to tune the threshold.

## 7. Runtime parity and leakage

Gold evidence, answers, and annotation ids are offline-only. Runtime computes
the same support and novelty features from the question, generated slots,
retrieved documents, and prior evidence history. Artifact metadata records the
adapter, utility provider, model versions, thresholds, split seed, and K
strategy.

Legacy logs without native ids may be joined to a uniquely matching dataset
annotation, but the join mode is recorded. If the annotation export also lacks
its native id, a stable SHA-256 identity is derived from its unique question
text and marked as derived rather than native. Closed candidate-pool paragraphs
receive stable question-scoped context ids; unmatched text receives only a
content-hash fallback and does not satisfy the corpus-id provenance gate.
Repeated trajectories for the same question id are always assigned to the same
split.

Artifacts and audits are strict JSON. Undefined metrics such as predictor
accuracy on a constant-K dataset are serialized as `null`, never `NaN` or
infinity.

SFT, calibration, GRPO, and final evaluation questions must be question-id
disjoint. Replaying a pretrained model's old HotpotQA knowledge is acceptable
for interface alignment, but it is not evidence of held-out generalization.

## 8. Version boundary

V5 is additive. V1-v4 artifacts, builders, and launchers remain unchanged.
The v5 artifact builder and audit must pass locally and on the remote pilot
before the live GRPO path is allowed to consume v5 artifacts.
