"""Cohort-specific settings loaded from a private JSON file.

Patient identifiers and per-patient data-handling rules are not stored in the
repository. They are read from a JSON file supplied with the private data; see
``configs/cohort_config.example.json`` for the expected keys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = "TEMP_NET_COHORT_CONFIG"

DEFAULTS = {
    # Patient groups that are always assigned to the test split.
    "locked_test_patients": [],
    # Exact lower-case patient-folder aliases merged into one patient group.
    "patient_aliases": {},
    # Free-text description of the alias rule, recorded in the split protocol.
    "patient_group_rule": "",
    # Substring replacements applied after canonicalising a patient name.
    "canonical_aliases": {},
    # Canonical row keys merged into one patient key when planning 3D cases.
    "row_aliases": {},
    # Patient -> supplementary tumor-mask filename (relative to the supplement folder).
    "supplementary_masks": {},
    # Patient -> the only session to which revised masks apply.
    "revision_mask_session_only": {},
    # Patients whose existing mask is combined with revised masks.
    "merge_existing_and_revision_masks": [],
    # Patients with two sessions encoded by a "_1" folder suffix (ses-01) or otherwise (ses-02).
    "session_by_suffix": [],
}


def load_cohort_config(path: str | os.PathLike | None = None, required: bool = False) -> dict:
    """Load the cohort config from ``path`` or the ``TEMP_NET_COHORT_CONFIG`` env var."""
    if path is None:
        path = os.environ.get(ENV_VAR) or None
    if path is None:
        if required:
            raise ValueError(f"Pass --cohort-config or set {ENV_VAR} to a private cohort config JSON")
        return dict(DEFAULTS)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown cohort config keys: {unknown}")
    return {**DEFAULTS, **payload}
