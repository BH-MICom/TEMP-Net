# Method implementation audit

The paper protocol is represented by the `scripts/paper_3d` and `train_paper_cv_fold.py` entry points. The implementation uses patient-level five-fold assignments, progressive short/full-horizon recurrent training, 180 virtual planes, optional MRT-plane holdout, reviewed probe centers, segment-specific measured baselines, and inclusive CEM43 thresholding. Inter-segment waiting dose is excluded from the paper protocol.

All trained checkpoints and patient data are external inputs. The repository contains no experiment results. The exact data split, center manifest, checkpoint, and reconstruction policy must be recorded by the caller when publishing a run.

The 3D runner records its plane count, holdout plane, baseline source, cooling policy, and reference-plane errors in output metadata.
