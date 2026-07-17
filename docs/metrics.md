# Metrics

KVCacheBench groups diagnostics by the level at which the compressed run can
diverge from FullCache.

- `CIF`: compression-induced failure indicator. It is 1 when FullCache is
  correct and the compressed run is wrong.
- `ERR`: Evidence Retention Rate. Fraction of annotated evidence tokens
  retained or represented by the compressed cache mapping.
- `ECov`: Evidence Coverage. Fraction of evidence spans with at least some
  retained/represented evidence.
- `DRR`: Distractor Retention Ratio. Distractor retention relative to evidence
  retention when distractor spans exist.
- `delta_NLL`: compressed gold-answer NLL minus full-cache gold-answer NLL.
  Larger values indicate stronger likelihood damage.
- `GPR`: Gold Probability Ratio, `P_compressed(gold) / P_full(gold)`.
  Smaller values indicate stronger loss of answer probability.
- `EAR`: Evidence Attention Ratio, available in the RULER-8K context-demand
  data where calibrated attention traces exist.

Cache metrics are not universally meaningful for every compression family.
Quantization can preserve all token positions while corrupting the value
representation; channel compression can retain token positions while changing
key/value geometry. The released rows therefore keep N/A values explicit and
pair cache metrics with logit and attention diagnostics.
