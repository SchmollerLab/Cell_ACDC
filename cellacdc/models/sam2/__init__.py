from cellacdc import myutils

# Check if SAM2 is installed
# Note: This assumes there's a check_install_sam2 function or we can adapt the existing one
try:
    import sam2
except ImportError:
    raise ImportError(
        "SAM2 is not installed. Please install it with: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
    )

import os
from pathlib import Path

# Get SAM2 models path
# Using the same pattern as segment_anything
_, sam_models_path = myutils.get_model_path('sam2', create_temp_dir=False)

# SAM2 model configurations
# Format: 'Display Name': ('config_file', 'checkpoint_filename')
model_types = {
    'Tiny': ('configs/sam2.1/sam2.1_hiera_t.yaml', 'sam2.1_hiera_tiny.pt'),
    'Small': ('configs/sam2.1/sam2.1_hiera_s.yaml', 'sam2.1_hiera_small.pt'),
    'Base Plus': ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
    'Large': ('configs/sam2.1/sam2.1_hiera_l.yaml', 'sam2.1_hiera_large.pt'),
}
