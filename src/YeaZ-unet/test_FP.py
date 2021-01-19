import sys
from skimage import io
from skimage.exposure import equalize_adapthist
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt
#append all the paths where the modules are stored. Such that this script
#looks into all of these folders when importing modules.
sys.path.append("./unet")
import neural_network as nn
from segment import segment

img = io.imread('test1.tif')

plt.imshow_tk(img)

img = equalize_adapthist(img)
img = img*1.0
pred = nn.prediction(img, True)

thresh = nn.threshold(pred)

lab = segment(thresh, pred, min_distance=5)

fig, ax = plt.subplots(1,2)

ax[0].imshow(img)
ax[1].imshow(lab)

plt.show()
