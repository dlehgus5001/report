"""
Unit tests for time-based pairing status assignment.

Verifies that:
  - "new"         → only objects from the LATER frame (current_capture_time)
  - "disappeared" → only objects from the EARLIER frame (past_capture_time)
  - "matched"     → objects present in BOTH frames

  - Time-ordering guard rejects inverted timestamps
  - Timezone-aware datetimes do NOT raise TypeError (offset-naive/aware mismatch)
  - Past detections with wrong timestamp are filtered out before matching
"""

import sys
import os
import uuid
import unittest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import patch

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pairing.temporal_pairing import pair_by_similarity, pair_by_tracking, _naive
from src.database.models import DetectionRecord, PairingRecord


# ── helpers ───────────────────────────────────────────────────────────────────

T_PAST    = datetime(2024, 1, 1, 6,  0, 0)   # 06:00 naive
T_CURRENT = datetime(2024, 1, 1, 12, 0, 0)   # 12:00 naive
T_FUTURE  = datetime(2024, 1, 1, 18, 0, 0)   # 18:00 naive


def _make_det_result(lat=10.0, lon=20.0, conf=0.9, cls="military tank"):
    """Minimal DetectionResult-like object (matches the dataclass fields used in pairing)."""
    return SimpleNamespace(
        detection_id=str(uuid.uuid4()),
        detection_time=T_CURRENT,
        image_id=str(uuid.uuid4()),
        object_class=cls,
        object_class_index=0,
        confidence=conf,
        bbox_x1=10.0, bbox_y1=10.0,
        bbox_x2=50.0, bbox_y2=50.0,
        lat=lat, lon=lon,
        source_type="satellite",
        mask_rle=None,
    )


def _make_det_record(lat=10.0, lon=20.0, conf=0.8, cls="military tank",
                     detection_time=None):
    """Minimal DetectionRecord ORM-like object."""
    rec = DetectionRecord(
        id=str(uuid.uuid4()),
        image_id=str(uuid.uuid4()),
        detection_time=detection_time or T_PAST,
        object_class=cls,
        object_class_index=0,
        confidence=conf,
        bbox_x1=10.0, bbox_y1=10.0,
        bbox_x2=50.0, bbox_y2=50.0,
        lat=lat, lon=lon,
        source_type="satellite",
        mask_rle=None,
    )
    return rec


def _make_tracked(past_id, lat=10.0, lon=20.0, score=0.9, cls="military tank"):
    """Minimal TrackedObject-like object."""
    return SimpleNamespace(
        past_detection_id=past_id,
        past_object_class=cls,
        score=score,
        bbox_x1=10.0, bbox_y1=10.0,
        bbox_x2=50.0, bbox_y2=50.0,
        lat=lat, lon=lon,
    )


def _statuses(records):
    return [r.status for r in records]


def _run_similarity(current_dets, past_dets,
                    current_time=T_CURRENT, past_time=T_PAST,
                    sim_matrix=None):
    """
    Run pair_by_similarity, injecting a fixed sim_matrix so CLIP is not needed.
    sim_matrix: 2-D list [[score_c0_p0, score_c0_p1, ...], ...]
    """
    import numpy as np
    from PIL import Image

    dummy_image = Image.new("RGB", (100, 100))

    if sim_matrix is not None:
        mat = np.array(sim_matrix, dtype=np.float32)
        with patch("src.pairing.temporal_pairing._compute_embeddings") as mock_emb:
            # _compute_embeddings is called twice (current, past); return row/col vectors
            mock_emb.side_effect = [
                np.eye(len(current_dets), mat.shape[1], dtype=np.float32),  # cur
                mat.T,  # past (shape M×D s.t. cur @ past.T == mat)
            ]
            # Easier: patch the sim matrix directly at the multiplication point
    # Simplest approach: patch _compute_embeddings to raise so fallback triggers,
    # then override COORDINATE_MATCH_RADIUS_DEG to make geo scores predictable.
    # OR: directly patch the matrix computation.

    # We patch _clip_embedder.embed to raise → exception caught → clip_sim_matrix=None
    # → geo-only fallback.  Geo score = 1 - dist/RADIUS.
    # By default COORDINATE_MATCH_RADIUS_DEG=0.01 (from config).
    with patch("src.pairing.temporal_pairing._clip_embedder") as mock_clip:
        mock_clip.embed.side_effect = RuntimeError("no CLIP in tests")
        if sim_matrix is not None:
            # Build fake embeddings whose dot-product equals the desired sim_matrix.
            import numpy as np
            mat = np.array(sim_matrix, dtype=np.float32)
            # Use the matrix directly by patching deeper
            mock_clip.embed.side_effect = None  # disable exception
            with patch("src.pairing.temporal_pairing._compute_embeddings") as mock_emb:
                N = len(current_dets)
                M = len(past_dets)
                D = max(N, M, 1)
                # Produce embeddings so cur @ past.T == mat
                # cur_emb[i] = standard basis e_i, past_emb[j] = mat[:,j] row
                cur_e = np.eye(N, D, dtype=np.float32)
                # past_emb s.t. cur_e @ past_emb.T == mat  → past_emb = mat.T transposed
                past_e = mat.T.copy()  # shape (M, N) — cur_e @ past_e.T = mat ✓
                mock_emb.side_effect = [cur_e, past_e]
                return pair_by_similarity(
                    current_detections=current_dets,
                    past_detections=past_dets,
                    current_image=dummy_image,
                    current_capture_time=current_time,
                    past_capture_time=past_time,
                    region_lat=10.0,
                    region_lon=20.0,
                    session_id="test-session",
                    source_type="satellite",
                )
        return pair_by_similarity(
            current_detections=current_dets,
            past_detections=past_dets,
            current_image=dummy_image,
            current_capture_time=current_time,
            past_capture_time=past_time,
            region_lat=10.0,
            region_lon=20.0,
            session_id="test-session",
            source_type="satellite",
        )


# ── test cases ────────────────────────────────────────────────────────────────

class TestNaiveHelper(unittest.TestCase):
    def test_naive_already_naive(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        self.assertIsNone(_naive(dt).tzinfo)

    def test_naive_strips_aware(self):
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _naive(dt)
        self.assertIsNone(result.tzinfo)
        self.assertEqual(result, datetime(2024, 1, 1, 12, 0, 0))

    def test_naive_none(self):
        self.assertIsNone(_naive(None))


class TestTimeGuard(unittest.TestCase):
    """Time-ordering guard: current must be STRICTLY later than past."""

    def test_time_inverted_returns_empty_similarity(self):
        cur = [_make_det_result()]
        past = [_make_det_record()]
        # past_time == current_time → should be rejected
        result = _run_similarity(cur, past,
                                 current_time=T_PAST, past_time=T_CURRENT)
        self.assertEqual(result, [], "Inverted timestamps should return []")

    def test_time_equal_returns_empty_similarity(self):
        cur = [_make_det_result()]
        past = [_make_det_record()]
        result = _run_similarity(cur, past,
                                 current_time=T_PAST, past_time=T_PAST)
        self.assertEqual(result, [], "Equal timestamps should return []")

    def test_time_inverted_returns_empty_tracking(self):
        past_rec = _make_det_record()
        tracked = [_make_tracked(past_rec.id)]
        result = pair_by_tracking(
            tracked_objects=tracked,
            current_detections=[_make_det_result()],
            past_detections=[past_rec],
            current_capture_time=T_PAST,   # earlier than past → inverted
            past_capture_time=T_CURRENT,
            region_lat=10.0, region_lon=20.0,
            session_id="test",
        )
        self.assertEqual(result, [], "Inverted timestamps should return []")

    def test_timezone_aware_does_not_raise(self):
        """offset-aware current_capture_time must not cause TypeError."""
        aware_current = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        naive_past    = datetime(2024, 1, 1,  6, 0, 0)   # naive from SQLite
        cur = [_make_det_result()]
        past = [_make_det_record(detection_time=naive_past)]
        try:
            result = _run_similarity(cur, past,
                                     current_time=aware_current,
                                     past_time=naive_past)
        except TypeError as e:
            self.fail(f"TypeError raised with mixed timezone datetimes: {e}")


class TestPastFilterByTimestamp(unittest.TestCase):
    """Past detections with detection_time > past_capture_time must be excluded."""

    def test_past_det_with_future_time_excluded_similarity(self):
        cur = [_make_det_result(lat=10.0, lon=20.0)]
        # This past detection has detection_time AFTER past_capture_time
        bad_past = _make_det_record(lat=10.0, lon=20.0,
                                    detection_time=T_FUTURE)  # T_FUTURE > T_PAST
        result = _run_similarity([cur[0]], [bad_past],
                                 current_time=T_CURRENT, past_time=T_PAST)
        # bad_past should be filtered out → cur becomes "new", bad_past ignored
        statuses = _statuses(result)
        self.assertIn("new", statuses)
        self.assertNotIn("disappeared", statuses,
                         "Past det with timestamp after past_capture_time must be excluded")
        self.assertNotIn("matched", statuses)

    def test_past_det_with_future_time_excluded_tracking(self):
        bad_past = _make_det_record(lat=10.0, lon=20.0, detection_time=T_FUTURE)
        tracked  = [_make_tracked(bad_past.id)]
        cur      = [_make_det_result(lat=10.0, lon=20.0)]
        result = pair_by_tracking(
            tracked_objects=tracked,
            current_detections=cur,
            past_detections=[bad_past],
            current_capture_time=T_CURRENT,
            past_capture_time=T_PAST,
            region_lat=10.0, region_lon=20.0,
            session_id="test",
        )
        statuses = _statuses(result)
        # bad_past filtered → past_by_id is empty → tracked gets past=None (still matched)
        # but the "disappeared" loop should not produce any disappeared records
        self.assertNotIn("disappeared", statuses,
                         "Past det with wrong timestamp should not appear as disappeared")


class TestNewStatus(unittest.TestCase):
    """new = object only in current (later) frame."""

    def test_all_new_when_no_past(self):
        cur = [_make_det_result(), _make_det_result()]
        result = _run_similarity(cur, [], current_time=T_CURRENT, past_time=None)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r.status == "new" for r in result))

    def test_new_has_current_capture_time(self):
        cur = [_make_det_result()]
        result = _run_similarity(cur, [], current_time=T_CURRENT, past_time=None)
        self.assertEqual(result[0].current_capture_time, T_CURRENT)
        self.assertIsNone(result[0].past_capture_time)

    def test_new_tracking_no_past(self):
        cur = [_make_det_result()]
        result = pair_by_tracking(
            tracked_objects=[],
            current_detections=cur,
            past_detections=[],
            current_capture_time=T_CURRENT,
            past_capture_time=None,
            region_lat=10.0, region_lon=20.0,
            session_id="test",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, "new")


class TestDisappearedStatus(unittest.TestCase):
    """disappeared = object only in past (earlier) frame."""

    def test_all_disappeared_when_no_current(self):
        past = [_make_det_record(), _make_det_record()]
        result = _run_similarity([], past, current_time=T_CURRENT, past_time=T_PAST)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r.status == "disappeared" for r in result))

    def test_disappeared_has_past_capture_time(self):
        past = [_make_det_record()]
        result = _run_similarity([], past, current_time=T_CURRENT, past_time=T_PAST)
        self.assertEqual(result[0].past_capture_time, T_PAST)
        self.assertIsNone(result[0].current_capture_time)

    def test_disappeared_tracking_no_tracked(self):
        past = [_make_det_record()]
        result = pair_by_tracking(
            tracked_objects=[],   # tracker found nothing → past object disappeared
            current_detections=[],
            past_detections=past,
            current_capture_time=T_CURRENT,
            past_capture_time=T_PAST,
            region_lat=10.0, region_lon=20.0,
            session_id="test",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, "disappeared")
        self.assertEqual(result[0].past_capture_time, T_PAST)


class TestMatchedStatus(unittest.TestCase):
    """matched = object in BOTH past and current frames (same position)."""

    def test_matched_via_tracking(self):
        past_rec = _make_det_record(lat=10.0, lon=20.0)
        cur_det  = _make_det_result(lat=10.0, lon=20.0)
        # Tracker confirms the past object is in the current frame (high IoU overlap)
        tracked  = [_make_tracked(past_rec.id)]
        result = pair_by_tracking(
            tracked_objects=tracked,
            current_detections=[cur_det],
            past_detections=[past_rec],
            current_capture_time=T_CURRENT,
            past_capture_time=T_PAST,
            region_lat=10.0, region_lon=20.0,
            session_id="test",
        )
        matched = [r for r in result if r.status == "matched"]
        self.assertEqual(len(matched), 1, "Should produce exactly one matched record")
        self.assertEqual(matched[0].current_capture_time, T_CURRENT)
        self.assertEqual(matched[0].past_capture_time, T_PAST)

    def test_matched_via_similarity_high_score(self):
        """High similarity (> SIMILARITY_MATCH_THRESHOLD) → matched."""
        cur  = [_make_det_result(lat=10.0, lon=20.0)]
        past = [_make_det_record(lat=10.0, lon=20.0)]
        # sim_matrix: [[0.9]] → above default threshold of 0.5
        result = _run_similarity(cur, past,
                                 current_time=T_CURRENT, past_time=T_PAST,
                                 sim_matrix=[[0.9]])
        matched = [r for r in result if r.status == "matched"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].current_capture_time, T_CURRENT)
        self.assertEqual(matched[0].past_capture_time, T_PAST)

    def test_no_match_low_similarity(self):
        """Low similarity (≤ SIMILARITY_MATCH_THRESHOLD) → new + disappeared."""
        cur  = [_make_det_result(lat=10.0, lon=20.0)]
        past = [_make_det_record(lat=10.0, lon=20.0)]
        # sim_matrix: [[0.1]] → below threshold → no match
        result = _run_similarity(cur, past,
                                 current_time=T_CURRENT, past_time=T_PAST,
                                 sim_matrix=[[0.1]])
        statuses = _statuses(result)
        self.assertIn("new", statuses)
        self.assertIn("disappeared", statuses)
        self.assertNotIn("matched", statuses)


class TestMixedScenario(unittest.TestCase):
    """Full scenario: 2 current, 2 past → 1 matched, 1 new, 1 disappeared."""

    def test_mixed_similarity(self):
        # cur0 ↔ past0: high similarity → matched
        # cur1: no good past match → new
        # past1: no good current match → disappeared
        cur  = [_make_det_result(lat=10.0, lon=20.0),
                _make_det_result(lat=50.0, lon=60.0)]
        past = [_make_det_record(lat=10.0, lon=20.0),
                _make_det_record(lat=90.0, lon=90.0)]
        # sim_matrix (2×2):
        #   cur0↔past0=0.9 (match), cur0↔past1=0.1
        #   cur1↔past0=0.1,         cur1↔past1=0.1 (both low → no match)
        sim = [[0.9, 0.1],
               [0.1, 0.1]]
        result = _run_similarity(cur, past,
                                 current_time=T_CURRENT, past_time=T_PAST,
                                 sim_matrix=sim)
        statuses = _statuses(result)
        self.assertIn("matched", statuses, "cur0↔past0 should be matched")
        self.assertIn("new", statuses,        "cur1 should be new")
        self.assertIn("disappeared", statuses, "past1 should be disappeared")

        # Verify time fields
        for r in result:
            if r.status in ("matched", "new"):
                self.assertEqual(r.current_capture_time, T_CURRENT)
            if r.status in ("matched", "disappeared"):
                self.assertEqual(r.past_capture_time, T_PAST)


class TestStatusExclusivity(unittest.TestCase):
    """No record should be ambiguously assigned to multiple statuses."""

    def test_no_duplicate_detection_ids_similarity(self):
        cur  = [_make_det_result() for _ in range(3)]
        past = [_make_det_record() for _ in range(3)]
        # All high similarity → all matched
        sim = [[0.9, 0.1, 0.1],
               [0.1, 0.9, 0.1],
               [0.1, 0.1, 0.9]]
        result = _run_similarity(cur, past,
                                 current_time=T_CURRENT, past_time=T_PAST,
                                 sim_matrix=sim)
        cur_ids  = [r.current_detection_id for r in result if r.current_detection_id]
        past_ids = [r.past_detection_id    for r in result if r.past_detection_id]
        self.assertEqual(len(cur_ids),  len(set(cur_ids)),  "Duplicate current_detection_id")
        self.assertEqual(len(past_ids), len(set(past_ids)), "Duplicate past_detection_id")


if __name__ == "__main__":
    unittest.main(verbosity=2)
