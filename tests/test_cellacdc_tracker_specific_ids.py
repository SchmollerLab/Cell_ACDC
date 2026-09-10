import numpy as np
from skimage.measure import regionprops

from cellacdc.regionprops import acdcRegionprops
from cellacdc.trackers.CellACDC import CellACDC_tracker
from cellacdc.trackers.CellACDC_2steps.CellACDC_2steps_tracker import tracker as TwoStepsTracker
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import tracker as NormalDivisionTracker


def test_calc_io_matrix_uses_regionprops_iteration_order_for_axes():
    prev_lab = np.array(
        [
            [1, 2],
            [7, 8],
        ],
        dtype=np.uint16,
    )
    lab = np.array(
        [
            [8, 1],
            [2, 7],
        ],
        dtype=np.uint16,
    )
    prev_rp = acdcRegionprops(prev_lab, precache_centroids=False)
    rp = acdcRegionprops(lab, precache_centroids=False)

    ioa_matrix, current_ids, previous_ids = CellACDC_tracker.calc_Io_matrix(
        lab, prev_lab, rp, prev_rp
    )
    old_ids, tracked_ids = CellACDC_tracker.assign(
        ioa_matrix, current_ids, previous_ids
    )

    assert current_ids == [obj.label for obj in rp]
    assert previous_ids == [obj.label for obj in prev_rp]
    assert dict(zip(old_ids, tracked_ids)) == {1: 2, 2: 7, 7: 8, 8: 1}


def test_track_frame_specific_ids_only_tracks_requested_current_ids():
    prev_lab = np.array(
        [
            [1, 1, 0, 5, 5],
            [1, 1, 0, 5, 5],
        ],
        dtype=np.uint16,
    )
    lab = np.array(
        [
            [7, 7, 0, 5, 5],
            [7, 7, 0, 5, 5],
        ],
        dtype=np.uint16,
    )

    tracked_lab, add_info = CellACDC_tracker.track_frame(
        prev_lab,
        regionprops(prev_lab),
        lab,
        regionprops(lab),
        IDs_curr_untracked=[7, 5],
        unique_ID=10,
        assign_unique_new_IDs=True,
        return_assignments=True,
        specific_IDs=[5],
    )

    np.testing.assert_array_equal(tracked_lab, lab)
    assert add_info['assignments'] == {}


def test_track_frame_specific_ids_skips_merging_with_unrelated_current_labels():
    prev_lab = np.array(
        [
            [5, 5, 0, 0],
            [5, 5, 0, 0],
        ],
        dtype=np.uint16,
    )
    lab = np.array(
        [
            [7, 7, 0, 5],
            [7, 7, 0, 5],
        ],
        dtype=np.uint16,
    )

    tracked_lab, add_info = CellACDC_tracker.track_frame(
        prev_lab,
        regionprops(prev_lab),
        lab,
        regionprops(lab),
        IDs_curr_untracked=[7, 5],
        unique_ID=10,
        assign_unique_new_IDs=True,
        return_assignments=True,
        specific_IDs=[7],
    )

    expected = np.array(
        [
            [10, 10, 0, 5],
            [10, 10, 0, 5],
        ],
        dtype=np.uint16,
    )

    np.testing.assert_array_equal(tracked_lab, expected)
    assert add_info['assignments'] == {7: 10}


def test_two_steps_specific_ids_can_match_selected_new_object_to_lost_previous_id():
    prev_lab = np.array(
        [
            [5, 5, 0, 0],
            [5, 5, 0, 0],
        ],
        dtype=np.uint16,
    )
    lab = np.array(
        [
            [7, 7, 0, 0],
            [7, 7, 0, 0],
        ],
        dtype=np.uint16,
    )

    tracked_lab, add_info = TwoStepsTracker(
        annotate_objects_tracked_second_step=False
    ).track_frame(
        prev_lab,
        lab,
        overlap_threshold=0.4,
        lost_IDs_search_range=10,
        unique_ID=10,
        return_assignments=True,
        specific_IDs=[7],
    )

    expected = np.array(
        [
            [5, 5, 0, 0],
            [5, 5, 0, 0],
        ],
        dtype=np.uint16,
    )

    np.testing.assert_array_equal(tracked_lab, expected)
    assert add_info['assignments'] == {7: 5}


def test_normal_division_specific_ids_preserve_division_context():
    prev_lab = np.array(
        [
            [5, 5, 5, 5],
            [5, 5, 5, 5],
        ],
        dtype=np.uint16,
    )
    lab = np.array(
        [
            [7, 7, 8, 8],
            [7, 7, 8, 8],
        ],
        dtype=np.uint16,
    )

    tracked_lab, add_info = NormalDivisionTracker().track_frame(
        prev_lab,
        lab,
        IoA_thresh=0.8,
        IoA_thresh_daughter=0.25,
        IoA_thresh_aggressive=0.5,
        min_daughter=2,
        max_daughter=2,
        unique_ID=20,
        return_assignments=True,
        specific_IDs=[7],
    )

    expected = np.array(
        [
            [20, 20, 8, 8],
            [20, 20, 8, 8],
        ],
        dtype=np.uint16,
    )

    np.testing.assert_array_equal(tracked_lab, expected)
    assert add_info['mothers'] == {5}
    assert add_info['assignments'] == {7: 20}


def test_normal_division_second_step_does_not_merge_existing_id():
    prev_lab = np.array(
        [
            [5, 5, 0, 0, 0, 0],
            [5, 5, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    lab = np.array(
        [
            [0, 0, 0, 7, 7, 5],
            [0, 0, 0, 7, 7, 5],
        ],
        dtype=np.uint16,
    )

    tracked_lab, add_info = NormalDivisionTracker().track_frame(
        prev_lab,
        lab,
        IoA_thresh=0.8,
        unique_ID=20,
        return_assignments=True,
        specific_IDs=[7],
    )

    expected = np.array(
        [
            [0, 0, 0, 20, 20, 5],
            [0, 0, 0, 20, 20, 5],
        ],
        dtype=np.uint16,
    )

    np.testing.assert_array_equal(tracked_lab, expected)
    assert add_info['assignments'] == {7: 20}


def test_normal_division_second_step_does_not_reuse_already_tracked_lost_id():
    # Regression test: tracking two newly drawn objects one at a time (as the
    # GUI does when the user draws missing cells individually) must not let
    # the second-step distance matching re-offer a lost ID that was already
    # assigned to the first object, even if it is the nearest candidate.
    prev_lab = np.array(
        [
            [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6],
            [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6],
        ],
        dtype=np.uint16,
    )

    tracker = NormalDivisionTracker()

    # First draw: raw ID 9, closest to lost ID 5.
    lab_draw1 = np.array(
        [
            [0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    tracked_lab_1, add_info_1 = tracker.track_frame(
        prev_lab,
        lab_draw1,
        IoA_thresh=0.8,
        lost_IDs_search_range=20,
        unique_ID=100,
        return_assignments=True,
        specific_IDs=[9],
    )
    assert add_info_1['assignments'] == {9: 5}

    # Second draw: raw ID 10, nearer to lost ID 5 (already used above) than
    # to the remaining lost ID 6, but 5 must not be available anymore.
    lab_draw2 = tracked_lab_1.copy()
    lab_draw2[:, 4:6] = 10
    tracked_lab_2, add_info_2 = tracker.track_frame(
        prev_lab,
        lab_draw2,
        IoA_thresh=0.8,
        lost_IDs_search_range=20,
        unique_ID=200,
        return_assignments=True,
        specific_IDs=[10],
    )

    expected = tracked_lab_1.copy()
    expected[:, 4:6] = 6

    np.testing.assert_array_equal(tracked_lab_2, expected)
    assert add_info_2['assignments'] == {10: 6}


def test_normal_division_first_step_does_not_merge_into_already_present_id():
    # Regression test: a newly drawn object that overlaps with a lost ID's
    # old position must not be tracked to that ID if the ID is already used
    # by another, untouched object elsewhere in the current frame (this used
    # to create a duplicate/merged label because `specific_IDs` restricted
    # the current IDs used for merge-avoidance to just the drawn object).
    # Instead, it should fall through to the 2nd step and match the actual
    # remaining lost ID, keeping `assignments` consistent with `tracked_lab`.
    prev_lab = np.array(
        [
            [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6],
            [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6],
        ],
        dtype=np.uint16,
    )
    # Raw ID 9 (newly drawn) overlaps lost ID 5's old position, but ID 5 is
    # already present, untouched, elsewhere in the current frame.
    lab = np.array(
        [
            [9, 9, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0],
            [9, 9, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )

    tracked_lab, add_info = NormalDivisionTracker().track_frame(
        prev_lab,
        lab,
        IoA_thresh=0.8,
        lost_IDs_search_range=20,
        unique_ID=100,
        return_assignments=True,
        specific_IDs=[9],
    )

    expected = np.array(
        [
            [6, 6, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0],
            [6, 6, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )

    np.testing.assert_array_equal(tracked_lab, expected)
    assert add_info['assignments'] == {9: 6}