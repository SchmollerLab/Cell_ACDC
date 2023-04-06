from cellacdc import myutils

myutils.check_install_segment_anything()

import os
from cellacdc import segment_anything_weights_filenames

_, sam_models_path = myutils.get_model_path('segment_anything', create_temp_dir=False)

model_types = {
    'Large': ('default', segment_anything_weights_filenames[0]), 
    'Medium': ('vit_l', segment_anything_weights_filenames[1]), 
    'Small': ('vit_b', segment_anything_weights_filenames[2])
}