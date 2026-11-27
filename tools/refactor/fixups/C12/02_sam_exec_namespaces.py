"""C12: the SAM/SAM2 embedding tests exec the model classes of
models/segment_anything/acdcSegment.py and models/sam2/acdcSegment.py in a
namespace that stands in for the plugin's module globals. The plugins read
myutils.to_uint8 as image_utils.to_uint8 after the rewrite, so the
namespaces provide image_utils (the module the plugins import) instead of
the myutils facade."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

SAM = 'tests/test_sam_embedding_reuse.py'
SAM2 = 'tests/test_sam2_embedding_reuse.py'


def main(repo):
    replace(repo, SAM,
            "from cellacdc import _types, myutils, sam_embeddings\n",
            "from cellacdc import _types, sam_embeddings\n"
            "from cellacdc.myutils import image_utils\n",
            items=[SAM + '::<imports>'])
    replace(repo, SAM,
            "        'np': np, 'pd': pd, 'cv2': cv2, 'myutils': myutils,"
            " '_types': _types,\n",
            "        'np': np, 'pd': pd, 'cv2': cv2, 'image_utils': image_utils,\n"
            "        '_types': _types,\n",
            items=[SAM + '::sam_wrapper'])
    replace(repo, SAM2,
            "from cellacdc import _types, myutils, sam_embeddings\n"
            "from cellacdc.myutils import model_modules\n",
            "from cellacdc import _types, sam_embeddings\n"
            "from cellacdc.myutils import image_utils, model_modules\n",
            items=[SAM2 + '::<imports>'])
    replace(repo, SAM2,
            "'defaultdict': defaultdict, 'myutils': myutils, '_types': _types,",
            "'defaultdict': defaultdict, 'image_utils': image_utils,\n"
            "        '_types': _types,",
            items=[SAM2 + '::sam2_wrapper'])


run(main)
