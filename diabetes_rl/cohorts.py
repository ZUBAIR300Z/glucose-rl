"""
Patient cohorts and train/test splits for the simglucose UVA/Padova population.

The simulator ships 30 virtual patients: 10 adolescents, 10 adults, 10 children.
Robust glucose control must generalize *across* this population, so we never
evaluate on a patient the agent trained on. These helpers define the cohorts and
produce held-out splits for that purpose.
"""
from __future__ import annotations

import random

ADOLESCENTS = [f"adolescent#{i:03d}" for i in range(1, 11)]
ADULTS = [f"adult#{i:03d}" for i in range(1, 11)]
CHILDREN = [f"child#{i:03d}" for i in range(1, 11)]
COHORTS = {"adolescent": ADOLESCENTS, "adult": ADULTS, "child": CHILDREN}
ALL_PATIENTS = ADOLESCENTS + ADULTS + CHILDREN  # 30 patients


def train_test_split(test_patients):
    """Explicit split: everything not in ``test_patients`` is training.

    Use this for leave-one-(or-few)-patient-out studies where you choose the
    held-out patients by name.
    """
    test = list(test_patients)
    unknown = [p for p in test if p not in ALL_PATIENTS]
    if unknown:
        raise ValueError(f"unknown patient(s): {unknown}")
    train = [p for p in ALL_PATIENTS if p not in test]
    return train, test


def stratified_holdout(n_per_cohort: int = 2, seed: int = 0):
    """Hold out ``n_per_cohort`` patients from EACH cohort as the test set.

    Stratifying keeps the held-out set representative of all three physiologies
    (e.g. n=2 -> 6 unseen test patients, 24 training patients). Returns
    ``(train_patients, test_patients)``.
    """
    rng = random.Random(seed)
    test = []
    for cohort in COHORTS.values():
        test.extend(rng.sample(cohort, n_per_cohort))
    train = [p for p in ALL_PATIENTS if p not in test]
    return train, test
