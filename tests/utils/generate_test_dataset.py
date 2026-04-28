"""Generate synthetic test dataset for GUI automation tests."""
import os
import numpy as np
from skimage import io
from pathlib import Path
import tempfile

def generate_test_dataset():
    """Create a minimal test dataset for GUI automation."""
    # Create a temporary directory for test data
    test_data_dir = Path("tests/test_data")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple multi-channel image stack
    # Simulate a real microscopy dataset
    height, width = 512, 512
    n_z = 10
    n_channels = 2
    
    print(f"Generating test dataset in {test_data_dir}")
    
    # Create a subdirectory for the dataset
    dataset_dir = test_data_dir / "test_sample"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic images for each channel and z-slice
    for ch in range(n_channels):
        for z in range(n_z):
            # Create synthetic image with some structure
            image = np.random.randint(0, 50, (height, width), dtype=np.uint16)
            
            # Add some bright spots (cells)
            n_cells = np.random.randint(3, 8)
            for _ in range(n_cells):
                cy, cx = np.random.randint(50, height-50, 2)
                radius = np.random.randint(10, 30)
                y, x = np.ogrid[:height, :width]
                mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                image[mask] = np.random.randint(500, 2000, dtype=np.uint16)
            
            # Create filename following common microscopy format
            filename = dataset_dir / f"img_t000_z{z:02d}_ch{ch:02d}.tif"
            io.imsave(str(filename), image)
            print(f"Created {filename.name}")
    
    print(f"\nTest dataset created at: {dataset_dir}")
    print(f"Created {n_z * n_channels} images")
    
    return str(dataset_dir)

if __name__ == "__main__":
    generate_test_dataset()
