import numpy as np
from skimage.measure import regionprops

from cellacdc.trackers.CellACDC import CellACDC_tracker
from cellacdc.trackers.CellACDC_2steps.CellACDC_2steps_tracker import tracker as TwoStepsTracker
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import tracker as NormalDivisionTracker


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

    tracked_lab, assignments = CellACDC_tracker.track_frame(
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
    assert assignments['assignments'] == {}


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