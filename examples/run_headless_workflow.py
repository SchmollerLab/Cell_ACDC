#!/usr/bin/env python3
"""Run Cell-ACDC segmentation and measurements without GUI or INI files.

Edit USER CONFIG below, then:

    python examples/run_headless_workflow.py

Data must follow the usual ACDC layout:

    /path/to/MyExperiment/Position_001/Images/phase.tif
    /path/to/MyExperiment/Position_001/Images/GFP.tif
    ...

You can also build or extend graphs directly — this script shows the
stock pipelines with plain Python configuration.
"""

from __future__ import annotations

import os
import sys

from tqdm import tqdm

# ---------------------------------------------------------------------------
# USER CONFIG — edit these
# ---------------------------------------------------------------------------

EXPERIMENT_PATH = "/path/to/MyExperiment"
USER_CH_NAME = "phase"
FLUOR_CHANNELS = ["GFP"]  # channels to quantify after segmentation
SEGM_ENDNAME = "segm.npz"  # output segm file basename
STOP_FRAME = 10  # frames to process per position (use 1 for single frame)

# Segmentation
MODEL_NAME = "cellpose"
MODEL_KWARGS = {"diameter": 30}
INIT_MODEL_KWARGS: dict = {}
DO_TRACKING = False
TRACKER_NAME = ""
TRACK_PARAMS: dict = {}
DO_POSTPROCESS = True
DO_SAVE = True
IS_SEGM_3D = False
USE_ROI = True

# Postprocess (empty dicts = defaults / no custom features)
STANDARD_POSTPROCESS_KWARGS: dict = {}
CUSTOM_POSTPROCESS_FEATURES: dict = {}
CUSTOM_POSTPROCESS_GROUPED_FEATURES: dict = {}

RUN_SEGMENTATION = True
RUN_MEASUREMENTS = True

# ---------------------------------------------------------------------------


def collect_position_paths(exp_path: str, user_ch: str) -> list[str]:
    from cellacdc import myutils

    paths: list[str] = []
    for pos in myutils.get_pos_foldernames(exp_path):
        images_path = os.path.join(exp_path, pos, "Images")
        paths.append(myutils.getChannelFilePath(images_path, user_ch))
    return paths


def build_segm_context():
    """Pure-Python workflow context — no kernel, no INI."""
    from cellacdc.workflow.state import WorkflowContext

    return WorkflowContext(
        user_ch_name=USER_CH_NAME,
        segm_endname=SEGM_ENDNAME,
        model_name=MODEL_NAME,
        tracker_name=TRACKER_NAME,
        do_tracking=DO_TRACKING,
        do_postprocess=DO_POSTPROCESS,
        do_save=DO_SAVE,
        is_segm_3d=IS_SEGM_3D,
        use_roi=USE_ROI,
        model_kwargs=dict(MODEL_KWARGS),
        init_model_kwargs=dict(INIT_MODEL_KWARGS),
        track_params=dict(TRACK_PARAMS),
        standard_postprocess_kwargs=dict(STANDARD_POSTPROCESS_KWARGS),
        custom_postprocess_features=dict(CUSTOM_POSTPROCESS_FEATURES),
        custom_postprocess_grouped_features=dict(CUSTOM_POSTPROCESS_GROUPED_FEATURES),
        size_t=STOP_FRAME,
        size_z=1,
    )


def run_segmentation(logger, log_path, paths: list[str]) -> None:
    from cellacdc.workflow.pipelines.batch import run_segm_batch
    from cellacdc.workflow.runnable import RunnableConfig

    ctx = build_segm_context()
    stops = [STOP_FRAME] * len(paths)
    pbar = tqdm(total=len(paths), desc="Segmentation", ncols=100)
    results = run_segm_batch(
        ctx,
        paths,
        stops,
        config=RunnableConfig(logger_func=logger.info),
        progress=pbar,
    )
    pbar.close()
    aborted = [r for r in results if getattr(r, "aborted", False)]
    if aborted:
        logger.warning(f"{len(aborted)} position(s) aborted during segmentation.")


def run_measurements(logger, log_path, paths: list[str]) -> None:
    from cellacdc import cli
    from cellacdc.workflow.adapters import configure_measurements_kernel_for_cli
    from cellacdc.workflow.pipelines.batch import run_measurements_batch
    from cellacdc.workflow.runnable import RunnableConfig

    kernel = cli.ComputeMeasurementsKernel(logger, log_path, is_cli=True)
    configure_measurements_kernel_for_cli(
        kernel,
        channels=[USER_CH_NAME, *FLUOR_CHANNELS],
        end_filename_segm=SEGM_ENDNAME.replace(".npz", ""),
        is_timelapse=STOP_FRAME > 1,
    )

    stops = [STOP_FRAME] * len(paths)
    pbar = tqdm(total=len(paths), desc="Measurements", ncols=100)
    run_measurements_batch(
        kernel,
        paths,
        stops,
        end_filename_segm=kernel.end_filename_segm,
        config=RunnableConfig(logger_func=logger.info),
        progress=pbar,
    )
    pbar.close()


def run_single_position_graph_example(path: str) -> None:
    """Minimal example: build one graph and invoke it once."""
    from cellacdc.workflow.pipelines.segm import build_position_segm_graph
    from cellacdc.workflow.runnable import RunnableConfig
    from cellacdc.workflow.state import PositionState

    graph = build_position_segm_graph(build_segm_context()).compile()
    result = graph.invoke(
        PositionState(img_path=path, stop_frame_n=STOP_FRAME),
        RunnableConfig(logger_func=print),
    )
    print("done:", result.aborted, result.error)


def main() -> int:
    if EXPERIMENT_PATH.startswith("/path/to"):
        print("Edit USER CONFIG in examples/run_headless_workflow.py first.", file=sys.stderr)
        return 1

    from cellacdc import myutils

    logger, _, log_path, _ = myutils.setupLogger(module="headless", logs_path=None)
    paths = collect_position_paths(EXPERIMENT_PATH, USER_CH_NAME)
    if not paths:
        logger.error(f"No positions found under {EXPERIMENT_PATH}")
        return 1

    logger.info(f"Found {len(paths)} position(s)")

    if RUN_SEGMENTATION:
        run_segmentation(logger, log_path, paths)

    if RUN_MEASUREMENTS:
        run_measurements(logger, log_path, paths)

    logger.info("Finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
