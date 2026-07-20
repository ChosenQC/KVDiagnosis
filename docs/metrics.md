# Metrics and Operational Signatures

KVDiagnosis treats FullCache as a paired control. Diagnostics describe where a
compressed run diverges; they do not identify a causal tensor, head, or channel
without an intervention.

## Endpoint and Transition Metrics

- `full_score`, `compressed_score`: task scorer outputs.
- `CIF`: one when FullCache is correct and compression is wrong.
- `score_drop`: FullCache score minus compressed score.

## Slot-Level Cache Metrics

For Qwen3-8B, a slot is one of 36 layers x 8 KV heads.

- `ERR_slot`: for each slot, compute the fraction of mapped evidence tokens
  retained, then average over slots.
- `ECov_slot`: fraction of slot/evidence-span pairs retaining at least 50% of
  that span.
- `ECov_slot_threshold`: fixed at 0.5.
- `coverage_type`: one of:
  - `measured_token_coverage`: retained original-token indices are observed;
  - `projected_token_coverage`: retained chunks are projected to their
    original-token spans;
  - `structural_position_addressability`: token positions remain addressable
    by construction, but retained representation fidelity is not measured;
  - `not_applicable`: no defensible token mapping is available.
- `ERR_slot_status`, `ECov_slot_status`: `available` or
  `not_applicable`.
- `structural_position_addressability`: explicit boolean that is never used
  as a numeric coverage surrogate.

No paper-facing file contains the deprecated cross-slot-union `ERR`, `ECov`,
or `DRR` fields. ThinK and QuantizedCache have structural position
addressability, so `ERR_slot` and `ECov_slot` are null and excluded from
coverage aggregates. For these methods, the supported statement is only:
annotated evidence positions remain addressable, while representation fidelity
is unknown.

## Logit Metrics

- `full_gold_NLL`, `compressed_gold_NLL`: paired gold-token negative
  log-likelihood.
- `delta_NLL`: compressed minus FullCache gold NLL.
- `GPR`: compressed-to-FullCache gold probability ratio.
- `KL`: paired predictive-distribution divergence.
- `TopK`: overlap/stability of top-token candidates.
- `gold_rank_shift`: change in gold-token rank when available.

## Attention Metrics

- `EAR`: evidence attention ratio from a valid eager-attention replay.
- `attention_available`: whether that replay preserves the method's semantic
  operation.
- `topk_available`: whether TopK was recorded.

Unavailable traces remain JSON `null`; they are never imputed. AdaKV and
ChunkKV projection coverage is not relabeled as native attention.

## Failure Signatures

The release uses observational names and frozen thresholds:

| Signature | Operational rule |
|---|---|
| `low_mapped_coverage` | measured/projected ECov_slot < 0.50 |
| `partial_mapped_coverage` | measured/projected 0.50 <= ECov_slot < 0.90 |
| `high_mapped_coverage_likelihood_drift` | measured/projected ECov_slot >= 0.90 and delta_NLL >= 1 |
| `structural_position_likelihood_drift` | positions are structurally addressable and delta_NLL >= 1 |
| `low_ear_candidate` | positions are mapped at ECov_slot >= 0.90 or structurally addressable, and a valid replay has EAR < 0.50 |
| `decode_scorer_candidate` | positions are mapped at ECov_slot >= 0.90 or structurally addressable, abs(delta_NLL) <= 0.10, and TopK >= 0.90 when available |
| `conflicting_diagnostic_signals` | more than one primitive rule fires |
| `ambiguous` | no frozen rule isolates a signature |

Counts are method-ratio rows. A source can appear multiple times, and changing
the thresholds can change bin membership. Numeric coverage rules apply only to
measured/projected rows. These signatures propose the next paired intervention;
they are not causal classes or deployment prevalence.
