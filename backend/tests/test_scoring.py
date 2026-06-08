"""Integration tests for scoring.process_frame_data.

Exercises the absent-is-not-zero contract, the audio-fallback behaviour, the
MC dropout cadence (per-connection state), and confirms the payload carries
frame_to_score_ms.

Real frame decoding is bypassed by feeding a pre-built frame through a stub
classifier so the test is fast and deterministic.
"""
from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest
import cv2

import ml.scoring as scoring_mod
from ml.ncnn.stream_state import FacialStreamState
from ml.ncnn_classifier import NCNNFacialPainClassifier


def _encode_jpeg(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class _StubNCNN(NCNNFacialPainClassifier):
    """Stand-in for NCNNFacialPainClassifier. Subclasses it so the predict
    signature matches and any type hints in scoring keep working, but skips
    the real __init__ so no model loads."""

    def __init__(self) -> None:
        # Intentionally do not call super().__init__: that would load the
        # real model and MediaPipe.
        self.calls: list[bool] = []
        self.next_result: dict | None = None
        self.face_detected_default = True

    def predict(self, frame, compute_uncertainty: bool) -> dict:
        self.calls.append(compute_uncertainty)
        if self.next_result is not None:
            return self.next_result
        if not self.face_detected_default:
            return {
                "face_detected": False,
                "prob_pain": None,
                "uncertainty": None,
                "facial_score": None,
                "landmarks": None,
                "frame_to_score_ms": 1.2,
            }
        return {
            "face_detected": True,
            "prob_pain": 0.6,
            "uncertainty": 0.05 if compute_uncertainty else None,
            "facial_score": 6.0,
            "landmarks": None,
            "frame_to_score_ms": 7.5,
        }

    def close(self) -> None:  # pragma: no cover - parity with real class
        pass


@pytest.fixture(autouse=True)
def _swap_facial_classifier(monkeypatch):
    """Install a stub classifier in place of the real singleton."""
    stub = _StubNCNN()
    monkeypatch.setattr(scoring_mod, "_facial_classifier", stub)
    monkeypatch.setattr(scoring_mod, "get_facial_classifier", lambda: stub)
    return stub


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def _frame_payload() -> dict:
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    return {"frame": _encode_jpeg(frame)}


@pytest.mark.integration
def test_face_detected_payload_carries_prob_uncertainty_and_latency(_swap_facial_classifier):
    state = FacialStreamState()
    result, new_state = _run(
        scoring_mod.process_frame_data(_frame_payload(), patient_id=1, stream_state=state)
    )
    assert result["face_detected"] is True
    assert result["prob_pain"] == 0.6
    assert result["uncertainty"] == 0.05  # MC ran on first frame
    assert result["frame_to_score_ms"] == 7.5
    assert new_state.frame_count == 1
    assert new_state.cached_uncertainty == 0.05
    # Smoother: first prob seeds the smoothed value at prob_pain.
    assert new_state.smoothed_prob_pain == 0.6
    # Facial-only fusion: 0.6 prob -> 6.0 score.
    assert result["facial_score"] == 6.0
    assert result["composite_score"] == 6.0
    assert result["signal_status"] == "facial_only"
    assert result["stale"] is False


@pytest.mark.integration
def test_absent_face_payload_has_none_for_prob_and_uncertainty(_swap_facial_classifier):
    _swap_facial_classifier.face_detected_default = False
    state = FacialStreamState()
    result, new_state = _run(
        scoring_mod.process_frame_data(_frame_payload(), patient_id=1, stream_state=state)
    )
    assert result["face_detected"] is False
    assert result["prob_pain"] is None
    assert result["uncertainty"] is None
    # The cached uncertainty stays None because no MC value was produced.
    assert new_state.cached_uncertainty is None
    assert new_state.frame_count == 1


@pytest.mark.integration
def test_cadence_skips_mc_between_refreshes(_swap_facial_classifier, monkeypatch):
    """With interval_frames=3, MC should run on call 0 and call 3 only."""
    from config import settings
    monkeypatch.setattr(settings, "ncnn_mc_interval_frames", 3)

    state = FacialStreamState()
    payload = _frame_payload()
    for _ in range(6):
        _, state = _run(scoring_mod.process_frame_data(payload, patient_id=1, stream_state=state))

    # The stub records whether compute_uncertainty=True for each call.
    # Expected: True at call indices 0 and 3, False otherwise.
    assert _swap_facial_classifier.calls == [True, False, False, True, False, False]


@pytest.mark.integration
def test_cached_uncertainty_is_reused_between_mc_refreshes(_swap_facial_classifier, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "ncnn_mc_interval_frames", 3)

    state = FacialStreamState()
    payload = _frame_payload()
    # Call 1: MC ran, uncertainty=0.05 cached.
    r1, state = _run(scoring_mod.process_frame_data(payload, patient_id=1, stream_state=state))
    # Call 2: cheap frame; uncertainty should be reported from cache.
    r2, state = _run(scoring_mod.process_frame_data(payload, patient_id=1, stream_state=state))
    # Call 3: cheap frame again.
    r3, state = _run(scoring_mod.process_frame_data(payload, patient_id=1, stream_state=state))

    assert r1["uncertainty"] == 0.05
    assert r2["uncertainty"] == 0.05
    assert r3["uncertainty"] == 0.05
    assert state.cached_uncertainty == 0.05


@pytest.mark.integration
def test_absent_face_falls_back_to_audio(_swap_facial_classifier):
    """Phase 4 interim: when facial is absent, fall back to audio so the
    composite has something to report. Phase 5 will formalize fusion."""
    _swap_facial_classifier.face_detected_default = False

    # Stub the cry analyzer too: bypass audio decoding, return a score.
    class _StubCry:
        def predict_from_bytes(self, _b):
            return {"cry_detected": True, "cry_type": "pain", "audio_score": 7.0}

    scoring_mod._cry_analyzer = _StubCry()  # type: ignore[assignment]
    try:
        state = FacialStreamState()
        payload = {"frame": _frame_payload()["frame"], "audio": base64.b64encode(b"x").decode()}
        result, _ = _run(scoring_mod.process_frame_data(payload, patient_id=1, stream_state=state))
        assert result["face_detected"] is False
        assert result["facial_score"] is None
        assert result["audio_score"] == 7.0
        assert result["composite_score"] == 7.0  # facial absent, audio carries
    finally:
        scoring_mod._cry_analyzer = None


@pytest.mark.integration
def test_none_payload_returns_safe_defaults():
    """A keep-alive with no data still returns a well-shaped dict."""
    state = FacialStreamState()
    result, new_state = _run(
        scoring_mod.process_frame_data(None, patient_id=1, stream_state=state)
    )
    assert result["face_detected"] is False
    assert result["prob_pain"] is None
    assert result["uncertainty"] is None
    assert result["composite_score"] is None
    assert result["signal_status"] == "unavailable"
    # State is not advanced when there is no data to process.
    assert new_state.frame_count == 0


@pytest.mark.integration
def test_smoother_holds_when_face_disappears(_swap_facial_classifier):
    """First frame seeds smoothed prob. Next frame loses the face; the
    smoothed value must hold rather than collapse toward zero."""
    state = FacialStreamState()
    r1, state = _run(scoring_mod.process_frame_data(_frame_payload(), 1, state))
    assert state.smoothed_prob_pain == 0.6

    # Drop the face on the next frame.
    _swap_facial_classifier.face_detected_default = False
    r2, state = _run(scoring_mod.process_frame_data(_frame_payload(), 1, state))
    # Smoother holds. Face-detected payload still reads None for live signals.
    assert state.smoothed_prob_pain == 0.6
    assert r2["face_detected"] is False
    assert r2["prob_pain"] is None


@pytest.mark.integration
def test_both_modalities_absent_holds_stale_composite(_swap_facial_classifier):
    """Frame 1 produces a fresh composite. Frame 2 loses face and has no
    audio: composite must be held tagged stale with an age, not faked to 0."""
    state = FacialStreamState()
    r1, state = _run(scoring_mod.process_frame_data(_frame_payload(), 1, state))
    assert r1["composite_score"] == 6.0
    assert r1["stale"] is False

    _swap_facial_classifier.face_detected_default = False
    r2, state = _run(scoring_mod.process_frame_data(_frame_payload(), 1, state))
    assert r2["composite_score"] == 6.0  # held
    assert r2["stale"] is True
    assert r2["signal_status"] == "stale"
    assert r2["alert_level"] == "stale"
    assert r2["stale_age_frames"] is not None and r2["stale_age_frames"] >= 1


@pytest.mark.integration
def test_both_absent_with_no_history_is_unavailable(_swap_facial_classifier):
    """Cold stream, no face and no audio: do not fabricate a composite."""
    _swap_facial_classifier.face_detected_default = False
    state = FacialStreamState()
    result, _ = _run(scoring_mod.process_frame_data(_frame_payload(), 1, state))
    assert result["composite_score"] is None
    assert result["signal_status"] == "unavailable"
    assert result["alert_level"] == "unavailable"
    assert result["pain_label"]["level"] == "Signal Unavailable"


@pytest.mark.integration
def test_stale_composite_flips_to_unavailable_past_the_cap(_swap_facial_classifier, monkeypatch):
    """Hold the stale composite up to max_stale_age_frames, then flip to
    unavailable. A minutes-old number rendered as if current is worse than
    an honest no-signal."""
    from config import settings
    monkeypatch.setattr(settings, "max_stale_age_frames", 3)

    state = FacialStreamState()
    payload = _frame_payload()

    # Frame 1: face detected, fresh composite cached.
    r1, state = _run(scoring_mod.process_frame_data(payload, 1, state))
    assert r1["composite_score"] == 6.0
    assert r1["signal_status"] == "facial_only"

    # Frames 2..4: face absent. Hold stale up to age = max_stale_age_frames.
    _swap_facial_classifier.face_detected_default = False
    for expected_age in (1, 2, 3):
        r, state = _run(scoring_mod.process_frame_data(payload, 1, state))
        assert r["signal_status"] == "stale", f"age={expected_age}"
        assert r["composite_score"] == 6.0
        assert r["stale_age_frames"] == expected_age

    # Frame 5: age = 4 > 3 cap. Flip to unavailable.
    r5, state = _run(scoring_mod.process_frame_data(payload, 1, state))
    assert r5["signal_status"] == "unavailable"
    assert r5["composite_score"] is None
    assert r5["stale"] is False
    assert r5["alert_level"] == "unavailable"


@pytest.mark.integration
def test_uncertainty_weighted_fusion_shifts_to_audio(_swap_facial_classifier):
    """High facial uncertainty should pull the composite toward the audio
    score when both are present."""
    # Configure the stub to return high uncertainty on the MC refresh frame.
    _swap_facial_classifier.next_result = {
        "face_detected": True,
        "prob_pain": 0.2,
        "uncertainty": 1.0,  # facial weight collapses to zero
        "facial_score": 2.0,
        "landmarks": None,
        "frame_to_score_ms": 5.0,
    }

    # Stub audio.
    class _StubCry:
        def predict_from_bytes(self, _b):
            return {"cry_detected": True, "cry_type": "pain", "audio_score": 9.0}

    scoring_mod._cry_analyzer = _StubCry()  # type: ignore[assignment]
    try:
        state = FacialStreamState()
        payload = {"frame": _frame_payload()["frame"], "audio": base64.b64encode(b"x").decode()}
        result, _ = _run(scoring_mod.process_frame_data(payload, 1, state))
        # Facial smoothed=0.2 -> score=2.0; uncertainty=1.0 zeros the facial
        # weight; composite must equal the audio score 9.0.
        assert result["composite_score"] == 9.0
        assert result["fusion_weights"]["facial"] == 0.0
        assert result["fusion_weights"]["audio"] == 1.0
    finally:
        scoring_mod._cry_analyzer = None


@pytest.mark.integration
def test_out_of_distribution_hard_stops_whole_composite(_swap_facial_classifier):
    """A non-infant face must stop the whole composite: no facial score and no
    audio score either, with a distinct signal_status. Even with a strong cry
    present, nothing is emitted, because a pain number attributed to the wrong
    subject is worse than none."""
    _swap_facial_classifier.next_result = {
        "face_detected": True,
        "prob_pain": None,
        "uncertainty": None,
        "facial_score": None,
        "landmarks": None,
        "frame_to_score_ms": 4.0,
        "out_of_distribution": True,
        "ood_reason": "subject_not_infant",
    }

    class _StubCry:
        def predict_from_bytes(self, _b):
            return {"cry_detected": True, "cry_type": "pain", "audio_score": 8.0}

    scoring_mod._cry_analyzer = _StubCry()  # type: ignore[assignment]
    try:
        state = FacialStreamState()
        payload = {"frame": _frame_payload()["frame"], "audio": base64.b64encode(b"x").decode()}
        result, new_state = _run(
            scoring_mod.process_frame_data(payload, patient_id=1, stream_state=state)
        )
        assert result["signal_status"] == "out_of_distribution"
        assert result["composite_score"] is None
        assert result["audio_score"] is None  # audio is not emitted under OOD
        assert result["facial_score"] is None
        assert result["ood_reason"] == "subject_not_infant"
        assert result["alert_level"] == "out_of_distribution"
        assert result["face_detected"] is True
        assert result["pain_label"]["level"] == "Subject Not Recognized"
        assert result["stale"] is False
        # State advanced for cadence, but no fresh composite was recorded.
        assert new_state.frame_count == 1
        assert new_state.last_composite_score is None
    finally:
        scoring_mod._cry_analyzer = None


@pytest.mark.integration
def test_out_of_distribution_does_not_pollute_the_smoother(_swap_facial_classifier):
    """An OOD frame between two valid frames must not drag the smoothed prob:
    the smoother holds its prior value across the OOD frame."""
    state = FacialStreamState()
    # Frame 1: valid face seeds the smoother at 0.6.
    _, state = _run(scoring_mod.process_frame_data(_frame_payload(), 1, state))
    assert state.smoothed_prob_pain == 0.6

    # Frame 2: OOD. Smoother must hold 0.6, not move toward zero.
    _swap_facial_classifier.next_result = {
        "face_detected": True,
        "prob_pain": None,
        "uncertainty": None,
        "facial_score": None,
        "landmarks": None,
        "frame_to_score_ms": 4.0,
        "out_of_distribution": True,
        "ood_reason": "subject_not_infant",
    }
    _, state = _run(scoring_mod.process_frame_data(_frame_payload(), 1, state))
    assert state.smoothed_prob_pain == 0.6
