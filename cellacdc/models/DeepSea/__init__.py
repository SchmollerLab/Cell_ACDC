import os

from cellacdc import myutils

myutils.check_install_package('deepsea')

import torch
import torchvision.transforms as transforms

_, deepsea_models_path = myutils.get_model_path('deepsea', create_temp_dir=False)

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
    _transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    model = DeepSeaClass(
        n_channels=1, n_classes=2, bilinear=True
    )
    model.load_state_dict(checkpoint)
    model = model.to(torch_device)
    return _transforms, torch_device, checkpoint, model
