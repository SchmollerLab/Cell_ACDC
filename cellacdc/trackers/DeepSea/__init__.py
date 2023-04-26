from cellacdc.models import DeepSea

from deepsea import tracker_transforms

import torchvision.transforms as transforms

image_size = [128,128]
image_means = [0.5]
image_stds = [0.5]

def _get_tracker_transforms():
    return tracker_transforms.Compose([
        tracker_transforms.Resize(image_size),
        tracker_transforms.Grayscale(num_output_channels=1),
        tracker_transforms.ToTensor(),
        tracker_transforms.Normalize(mean=image_means, std=image_stds)
    ])