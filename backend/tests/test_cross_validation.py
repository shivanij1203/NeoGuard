"""Tests for the subject-wise cross-validation splitter.

The one invariant that matters: no subject_id may appear in more than one fold.
These tests assert that invariant directly on synthetic subject IDs, plus the
supporting properties (full coverage, exactly-once placement, determinism) and
the empty and single-subject edge cases.
"""
from __future__ import annotations

import pytest

from ml.ncnn.cross_validation import (
    group_indices_by_subject,
    subject_wise_kfold,
    subject_wise_splits,
)


def _subject_of_index(subject_ids, fold):
    """Return the set of subject_ids touched by the sample indices in a fold."""
    return {subject_ids[index] for index in fold}


@pytest.mark.unit
def test_no_subject_appears_in_more_than_one_fold():
    # Ten subjects, several samples each, deliberately interleaved so a naive
    # sample-level split would scatter a subject across folds.
    subject_ids = [f"subj_{i % 10}" for i in range(100)]
    folds = subject_wise_kfold(subject_ids, n_splits=5)

    seen_subjects: set[str] = set()
    for fold in folds:
        fold_subjects = _subject_of_index(subject_ids, fold)
        # No subject in this fold may have appeared in any earlier fold.
        assert seen_subjects.isdisjoint(fold_subjects)
        seen_subjects |= fold_subjects


@pytest.mark.unit
def test_every_sample_lands_in_exactly_one_fold():
    subject_ids = [f"subj_{i % 7}" for i in range(70)]
    folds = subject_wise_kfold(subject_ids, n_splits=3)

    all_indices = [index for fold in folds for index in fold]
    # Exactly once: no duplicates, and full coverage of every input index.
    assert sorted(all_indices) == list(range(len(subject_ids)))
    assert len(all_indices) == len(set(all_indices))


@pytest.mark.unit
def test_all_samples_of_a_subject_share_a_fold():
    subject_ids = [f"subj_{i % 4}" for i in range(40)]
    folds = subject_wise_kfold(subject_ids, n_splits=4)

    # Build a reverse map: which fold did each subject land in.
    subject_to_fold: dict[str, int] = {}
    for fold_index, fold in enumerate(folds):
        for index in fold:
            subject = subject_ids[index]
            if subject in subject_to_fold:
                assert subject_to_fold[subject] == fold_index
            else:
                subject_to_fold[subject] = fold_index


@pytest.mark.unit
def test_split_is_deterministic():
    subject_ids = [f"subj_{i % 9}" for i in range(81)]
    first = subject_wise_kfold(subject_ids, n_splits=4)
    second = subject_wise_kfold(subject_ids, n_splits=4)
    assert first == second


@pytest.mark.unit
def test_empty_input_returns_empty_folds():
    folds = subject_wise_kfold([], n_splits=5)
    assert folds == [[], [], [], [], []]


@pytest.mark.unit
def test_single_subject_lands_in_one_fold_others_empty():
    subject_ids = ["only_subject"] * 12
    folds = subject_wise_kfold(subject_ids, n_splits=4)

    non_empty = [fold for fold in folds if fold]
    # A single subject cannot be split without leaking it, so exactly one fold
    # holds all its samples and the rest stay empty.
    assert len(non_empty) == 1
    assert sorted(non_empty[0]) == list(range(12))


@pytest.mark.unit
def test_fewer_subjects_than_folds_leaves_folds_empty_without_leaking():
    subject_ids = ["a"] * 3 + ["b"] * 3
    folds = subject_wise_kfold(subject_ids, n_splits=5)

    non_empty = [fold for fold in folds if fold]
    assert len(non_empty) == 2
    # Even when folds outnumber data, the subject disjointness invariant holds.
    seen: set[str] = set()
    for fold in folds:
        fold_subjects = _subject_of_index(subject_ids, fold)
        assert seen.isdisjoint(fold_subjects)
        seen |= fold_subjects


@pytest.mark.unit
def test_n_splits_below_one_raises():
    with pytest.raises(ValueError):
        subject_wise_kfold(["a", "b"], n_splits=0)


@pytest.mark.unit
def test_none_subject_id_raises():
    with pytest.raises(ValueError):
        subject_wise_kfold(["a", None, "b"], n_splits=2)


@pytest.mark.unit
def test_group_indices_by_subject_groups_correctly():
    subject_ids = ["x", "y", "x", "z", "y", "x"]
    groups = group_indices_by_subject(subject_ids)
    assert groups == {"x": [0, 2, 5], "y": [1, 4], "z": [3]}


@pytest.mark.unit
def test_splits_train_and_test_are_subject_disjoint():
    subject_ids = [f"subj_{i % 6}" for i in range(60)]
    splits = subject_wise_splits(subject_ids, n_splits=3)

    assert len(splits) == 3
    for train_indices, test_indices in splits:
        # Indices never overlap.
        assert set(train_indices).isdisjoint(test_indices)
        # Subjects never overlap either, which is the property the harness needs.
        train_subjects = _subject_of_index(subject_ids, train_indices)
        test_subjects = _subject_of_index(subject_ids, test_indices)
        assert train_subjects.isdisjoint(test_subjects)
        # Train plus test covers every sample exactly once.
        assert sorted(train_indices + test_indices) == list(range(len(subject_ids)))


@pytest.mark.unit
def test_subject_counts_are_balanced_across_folds():
    # Equal-sized subjects over a divisible split should balance perfectly.
    subject_ids = [f"subj_{i % 8}" for i in range(80)]
    folds = subject_wise_kfold(subject_ids, n_splits=4)
    sizes = [len(fold) for fold in folds]
    assert sizes == [20, 20, 20, 20]
