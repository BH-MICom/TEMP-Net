# TEMP-Net

TEMP-Net is a treatment-conditioned recurrent temperature-prediction framework for MR-guided laser interstitial thermal therapy. The repository contains only source code, configuration examples, and method documentation. Patient data, experiment outputs, and model weights are supplied separately.

## Method

The model uses T2-weighted structural input, a baseline temperature map, a tumor mask, a fiber-centered Gaussian field, and a compatibility channel. Power, fiber coordinates, and treatment time are encoded as global conditions and injected with FiLM. Temperature evolution is propagated through coarse and local recurrent feature states. The decoder predicts a baseline-referenced temperature residual.

Probe centers are reviewed inputs. The NPZ preparation script reads them from a center manifest and uses the same centers for the Gaussian field, coordinate condition, and physical reconstruction. See `docs/CENTER_PROVENANCE.md`.

## Repository layout

```text
src/temp_net/                 Model, data loader, and reconstruction modules
scripts/                      Data preparation, training, evaluation, and paper 3D entry points
scripts/paper_3d/             180-plane reconstruction and optional MRT-plane holdout
docs/                         Method, metric, center, and release documentation
configs/                      Configuration examples
```

## Environment

Use Conda with Python 3.10 and a CUDA-compatible PyTorch build.

```bash
conda env create -f environment.yml
conda activate temp-net
pip install -e .
pip install -r requirements-3d.txt
```

Check the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Training-ready data

The training directory must contain `manifests/segment_manifest.csv`, `normalization.json`, `dataset_summary.json`, and the referenced NPZ files. NPZ preparation requires a reviewed center manifest.

```bash
python scripts/prepare_training_npz.py \
  --dataset-root /path/to/LITT_2D_4PLANE_DATASET \
  --output /path/to/training_ready \
  --center-manifest /path/to/reviewed_centers.json   --cohort-config /path/to/private/cohort_config.json
```

The center manifest can be JSON:

```json
{
  "B01_Patient01_TrajA_P1_S001": {"center_rc": [253.0, 247.0]}
}
```

## Private cohort config

Patient identifiers and per-patient data-handling rules (locked test patients, folder aliases, mask revisions) are not part of this repository. Supply them in a private JSON file that stays with the data, using the keys in `configs/cohort_config.example.json`, and pass it with `--cohort-config` or the `TEMP_NET_COHORT_CONFIG` environment variable. `prepare_training_npz.py` and `finalize_training_npz.py` require it; `scripts/paper_3d/prepare_cohort.py` reads it from the environment variable.

## Paper training protocol

Create patient-level folds:

```bash
python scripts/make_patient_folds.py \
  --data /path/to/training_ready \
  --output /path/to/patient_folds.json \
  --folds 5
```

Train one fold with progressive recurrent training:

```bash
python scripts/train_paper_cv_fold.py \
  --data /path/to/training_ready \
  --folds /path/to/patient_folds.json \
  --fold 0 \
  --output /path/to/runs/fold-00 \
  --short-horizon 16 \
  --stage1-epochs 30 \
  --stage2-epochs 40
```

Run all five folds separately and combine their held-out predictions for out-of-fold evaluation.

## Paper 3D protocol

The paper 3D path uses 180 planes at 1-degree spacing and supports optional monitored-plane holdout.

```bash
export TEMP_NET_DATA_ROOT=/path/to/data_root
export TEMP_NET_WEIGHTS=/path/to/fold_checkpoint/model_stage2_full_best.pt
python scripts/paper_3d/prepare_cohort.py \
  --archive /path/to/segmented_records.zip \
  --output /path/to/paper_3d_run \
  --num-planes 180 \
  --holdout-plane 1
```

See `scripts/paper_3d/README.md` for the required registered MRI, masks, geometry, and cooling inputs.

## Evaluation metrics

The evaluation definitions are documented in `docs/EVALUATION_METRICS.md`. They include MAE, RMSE, Pearson correlation, SSIM, Dice, HD95, ASSD, precision, recall, and inclusive CEM43 thresholds. Thresholds are evaluated as `CEM43 >= 2`, `CEM43 >= 10`, and `CEM43 >= 60` minutes.
## Cross-validation data flow

For each held-out fold, the held-out patients are used only once for final evaluation. The remaining four folds are randomly split at the patient level into an internal training set and an internal validation set. Stage I and Stage II checkpoint selection use only the internal validation patients. The held-out fold is never used for checkpoint selection.
