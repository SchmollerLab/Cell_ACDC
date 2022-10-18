try:
    from cellpose.models import MODEL_NAMES
    CELLPOSE_MODELS = MODEL_NAMES
except Exception as e:
    CELLPOSE_MODELS = [
        'cyto','nuclei','tissuenet','livecell', 'cyto2',
        'CP', 'CPx', 'TN1', 'TN2', 'TN3', 'LC1', 'LC2', 'LC3', 'LC4'
    ]

STARDIST_MODELS = [
    '2D_versatile_fluo',
    '2D_versatile_he',
    '2D_paper_dsb2018'
]

try:
    from omnipose.core import OMNI_MODELS
except Exception as e:
    OMNI_MODELS = []