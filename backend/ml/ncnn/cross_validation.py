"""Subject-wise cross-validation splitting for the N-CNN training harness.

The single hard rule of this project's evaluation is that no subject may appear
in more than one fold. A frame-level or sample-level split leaks a subject's
appearance, lighting, and skin tone across train and test, so the model can
score well by recognising the baby rather than the pain. Any split that puts the
same subject on both sides is a bug, here and everywhere else in the pipeline.

This module is a pure splitter. It takes samples tagged with a subject_id and
returns folds of sample indices, partitioned so that every sample sharing a
subject_id lands in the same fold. It does not load data, train, or evaluate.

TODO: the training and calibration-fit harness plugs in here once USF-MNPAD-I
access is granted. The harness must build each train fold from the union of all
other folds, fit the N-CNN on it, then fit the temperature scaler on a
subject-disjoint held-out fold (see ml.ncnn.calibration.TemperatureScaler.fit).
Until that path runs against real labelled NICU data, no metric can be claimed.

TODO: stratification. This splitter balances folds by sample count only. It
does not balance the pain vs no-pain label ratio across folds, and with few
NICU subjects an unlucky split can land most pain-positive subjects in one
fold. Once real labels exist, add subject-wise stratification: keep the
no-subject-in-two-folds invariant as the hard constraint, and within it bin
subjects by their pain-positive rate so each fold sees a comparable label
mix. Stratify on the subject summary, never on individual frames, or the
subject-disjointness above breaks.

Research prototype, not a medical device.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence


def group_indices_by_subject(
    subject_ids: Sequence[Hashable],
) -> dict[Hashable, list[int]]:
    """Map each subject_id to the sorted list of sample indices that carry it.

    Returns a new dict; the input sequence is not mutated. Insertion order
    follows first appearance of each subject_id, which keeps the grouping
    deterministic for a given input.
    """
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, subject_id in enumerate(subject_ids):
        if subject_id is None:
            raise ValueError(
                f"sample at index {index} has subject_id None; every sample must "
                "carry a subject_id for a subject-wise split"
            )
        groups[subject_id].append(index)
    # Convert back to a plain dict so callers cannot accidentally create empty
    # groups by reading a missing key.
    return {subject_id: indices for subject_id, indices in groups.items()}


def subject_wise_kfold(
    subject_ids: Sequence[Hashable],
    n_splits: int,
) -> list[list[int]]:
    """Partition sample indices into ``n_splits`` folds with no subject overlap.

    Each returned fold is a list of sample indices intended as the test fold for
    one cross-validation round; the train fold is the union of all other folds.
    Every sample lands in exactly one fold, and all samples sharing a subject_id
    land in the same fold.

    Subjects are assigned greedily, largest first, each to the currently
    smallest fold, which balances sample counts across folds without any
    randomness. Ties break by fold index then by subject_id, so the split is
    fully deterministic for a given input and ``n_splits``.

    Edge cases:
      - An empty input returns ``n_splits`` empty folds.
      - Fewer subjects than ``n_splits`` leaves some folds empty rather than
        splitting a subject. An empty test fold is honest; a leaked subject is
        not.

    Raises:
      - ValueError if ``n_splits`` is less than 1, or if any subject_id is None.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}")

    groups = group_indices_by_subject(subject_ids)

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    # Largest subjects first so the greedy balance has the most room to even out.
    # Sort by descending sample count, then by subject_id repr for a stable order
    # that does not depend on dict iteration details across runs.
    ordered_subjects = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), repr(item[0])),
    )

    for subject_id, indices in ordered_subjects:
        # Place this subject's whole block in the currently smallest fold.
        target = min(range(n_splits), key=lambda f: (fold_sizes[f], f))
        folds[target].extend(indices)
        fold_sizes[target] += len(indices)

    # Keep each fold's indices in ascending order for readable, stable output.
    return [sorted(fold) for fold in folds]


def subject_wise_splits(
    subject_ids: Sequence[Hashable],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    """Yield ``(train_indices, test_indices)`` pairs for each fold.

    Thin convenience wrapper over :func:`subject_wise_kfold` for the future
    training harness: the test fold is one fold, the train fold is the sorted
    union of all other folds. Subject disjointness between train and test is
    inherited directly from the underlying partition.
    """
    folds = subject_wise_kfold(subject_ids, n_splits)
    splits: list[tuple[list[int], list[int]]] = []
    for test_fold_index in range(n_splits):
        test_indices = folds[test_fold_index]
        train_indices = sorted(
            index
            for other_index, fold in enumerate(folds)
            if other_index != test_fold_index
            for index in fold
        )
        splits.append((train_indices, test_indices))
    return splits
