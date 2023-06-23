import numpy as np

class SizesToResize:
    values = np.arange(256, 1025, 128)

class tracker:
    def __init__(self, gpu=False):
        pass
    
    def track(
            self, segm_video, image, 
            resize_to_square_with_size: SizesToResize=256,
            signals=None
        ):
        return segm_video

def url_help():
    return 'https://deepmind-tapir.github.io/'