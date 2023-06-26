import numpy as np
import jax
import tree

from tapnet import tapir_model

def build_model(frames, query_points):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR()
    outputs = model(
        video=frames,
        is_training=False,
        query_points=query_points,
        query_chunk_size=64,
    )
    return outputs

def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
        frames: [num_frames, height, width, 3], [0, 255], float

    Returns:
        frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
        occlusions: [num_points, num_frames], [-inf, inf], np.float32
        expected_dist: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
        visibles: [num_points, num_frames], bool
    """
    # visibles = occlusions < 0
    visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
    return visibles

def inference(frames, query_points, model_apply, params, state):
    """Inference on one video.

    Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8
        query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
        tracks: [num_points, 3], [-1, 1], [t, y, x]
        visibles: [num_points, num_frames], bool
    """
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    tracks, occlusions, expected_dist = (
        outputs['tracks'], outputs['occlusion'], outputs['expected_dist']
    )

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles