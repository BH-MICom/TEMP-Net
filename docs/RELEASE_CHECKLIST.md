# Code release checklist

- [x] Source code is separated from patient data, weights, and run outputs.
- [x] Patient-level five-fold assignment is available.
- [x] Two-stage short/full-horizon training is available.
- [x] The paper 3D path uses 180 virtual planes and optional MRT-plane holdout.
- [x] Reviewed center manifests are required for NPZ preparation.
- [x] CEM43 comparisons use inclusive `>=` thresholds.
- [ ] Run all five folds and publish their out-of-fold metrics.
- [ ] Run the full paper 3D cohort with fold-specific weights.
