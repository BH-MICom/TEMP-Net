# Paper-protocol 3D reconstruction

This directory is the paper-aligned 3D path. It differs from the legacy cohort runner in the repository root.

The paper configuration uses 180 virtual planes over 0 to 180 degrees at 1-degree spacing. The optional `--holdout-plane 1` or `--holdout-plane 2` removes one measured MRT plane from baseline estimation and anchor inputs while retaining it for reference-plane evaluation.

Prepare a plan:

```bash
export TEMP_NET_DATA_ROOT=/path/to/data_root
export TEMP_NET_WEIGHTS=/path/to/fold_checkpoint/model_stage2_full_best.pt
python scripts/paper_3d/prepare_cohort.py \
  --archive /path/to/segmented_records.zip \
  --output /path/to/paper_3d_run \
  --num-planes 180 \
  --holdout-plane 1
```

Run a case with the paper runner:

```bash
python scripts/paper_3d/run_cohort_anchor.py \
  --output /path/to/paper_3d_run \
  --case PATIENT/ses-01
```

The runner records the holdout plane, plane count, segment-specific baseline source, and reference-plane errors in each segment metadata file. Each segment starts from its measured baseline field. Inter-segment waiting dose is excluded from the paper protocol.

Set `TEMP_NET_COHORT_CONFIG` to the private cohort config described in the main README. Set `TEMP_NET_DATA_ROOT` to the directory containing the registered MNI dataset and training-ready records. Set `TEMP_NET_MODEL_SRC` to the repository `src` directory when the script is launched outside the repository root. The plan's `weights` field must point to a checkpoint produced by `train_paper_cv_fold.py` for the held-out patient fold.
