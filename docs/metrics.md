# Metrics and Operational Signatures

KVCacheBench treats FullCache as a paired control. Diagnostics describe where a
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
- `retention_semantics`: either measured per-slot original-position mapping or
  structural preservation of all token positions.

No paper-facing file contains the deprecated cross-slot-union `ERR`, `ECov`,
or `DRR` fields. ThinK and QuantizedCache have structural position coverage;
a value of one does not imply that key/value representations are preserved.

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
| `low_slot_coverage` | ECov_slot < 0.50 |
| `partial_slot_coverage` | 0.50 <= ECov_slot < 0.90 |
| `high_coverage_likelihood_drift` | ECov_slot >= 0.90 and delta_NLL >= 1 |
| `low_ear_candidate` | high coverage and EAR < 0.50 without another retained-state flag |
| `decode_scorer_candidate` | high coverage, abs(delta_NLL) <= 0.10, and TopK >= 0.90 when available |
| `conflicting_retained_signals` | more than one primitive rule fires |
| `ambiguous` | no frozen rule isolates a signature |

Counts are method-ratio rows. A source can appear multiple times, and changing
the thresholds can change bin membership. These signatures propose the next
paired intervention; they are not causal classes or deployment prevalence.
