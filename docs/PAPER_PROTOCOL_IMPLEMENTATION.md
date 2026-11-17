# Paper protocol implementation

The paper path implements patient-level five-fold assignment, two-stage short/full-horizon training, 180 virtual planes, optional monitored-plane holdout, reviewed probe centers, and inclusive CEM43 thresholds. Each execution writes its own fold, center manifest, checkpoint, and reconstruction policy metadata.

The repository contains source code only. Experiment outputs and checkpoints are supplied by the caller and are never bundled into the release.
The fold trainer creates an internal patient-level training/validation split from the four non-held-out folds. The held-out fold is evaluated only after Stage II checkpoint selection. Its metrics are written under `heldout_test` in `final_metrics.json`.
