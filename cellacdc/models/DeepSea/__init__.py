import os

import numpy as np

from cellacdc import myutils

myutils.check_install_package('deepsea')

import torch
import torchvision.transforms as transforms

from PIL import Image

_, deepsea_models_path = myutils.get_model_path('deepsea', create_temp_dir=False)

image_size = [383,512]
image_means = [0.0]
image_stds = [1.0]

def _get_transforms(image_size, image_means, image_stds):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

def _init_model(
        checkpoint_filename, DeepSeaClass, image_size, image_means, 
        image_stds, gpu=False
    ):
     # Initialize torch device
    if gpu:
        from cellacdc import is_mac
        import platform
        cpu = platform.processor()
        if is_mac and cpu == 'arm':
            device = 'cpu'
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    torch_device = torch.device(device)

    # Initialize checkpoint
    checkpoint_path = os.path.join(deepsea_models_path, checkpoint_filename)
    checkpoint = torch.load(checkpoint_path)

    # Initialize model
    _transforms = _get_transforms(image_size, image_means, image_stds)

    model = DeepSeaClass(
        n_channels=1, n_classes=2, bilinear=True
    )
    model.load_state_dict(checkpoint)
    model = model.to(torch_device)
    return _transforms, torch_device, checkpoint, model

def _resize_img(img: Image.Image, image_size, image_means, image_stds, device):
    _transforms = _get_transforms(image_size, image_means, image_stds)
    tensor_img = _transforms(img).to(device=device, dtype=torch.float32)
    resized_img = tensor_img.cpu().numpy()[0,:,:]
    img_min = np.min(resized_img)
    img_max = np.max(resized_img)
    img_range = img_max - img_min
    resized_img = (resized_img - img_min) / img_range * 255
    return resized_img
