import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

import skimage.draw
import scipy.interpolate

import matplotlib.pyplot as plt

pwd_path = os.path.dirname(os.path.abspath(__file__))

img = np.zeros((1000,1000), dtype=np.uint8)

dfs = []
keys = []

space_size_step = 10
square_side_range_min, square_side_range_max = 10, 600

for space_size in tqdm(np.arange(10,1001,10), ncols=100):
    bbox_areas = []
    exec_times = []
    space = np.linspace(0, 1, space_size)
    for side in tqdm(np.arange(square_side_range_min,square_side_range_min+1,2), ncols=100):
        img[:] = 0

        half_side = int(side/2)

        left = 500-half_side
        right = 500+half_side
        
        anchors_xx = [left,right,right,left,left]
        anchors_yy = [left,left,right,right,left]

        bbox_area = side**2
        bbox_areas.append(bbox_area)

        tck, u = scipy.interpolate.splprep( 
            [anchors_xx, anchors_yy], s=0, k=3, per=False
        )
        xi, yi = scipy.interpolate.splev(space, tck)

        t0 = time.perf_counter()
        rr, cc = skimage.draw.polygon(yi, xi, shape=img.shape)
        t1 = time.perf_counter()

        exec_times.append((t1-t0)*1000)

        img[rr, cc] = 2

        t0 = time.perf_counter()
        rr, cc = skimage.draw.polygon(anchors_yy, anchors_xx, shape=img.shape)
        t1 = time.perf_counter()

        img[rr, cc] = 1

    df = pd.DataFrame({'bbox_area': bbox_areas, 'exec_time': exec_times}).set_index('bbox_area')
    dfs.append(df)
    keys.append(space_size)

final_df = pd.concat(dfs, keys=keys, names=['space_size', 'bbox_area'])

df_filename = (
    f'side_range_{square_side_range_min}-{square_side_range_max}_'
    f'space_size_step_{space_size_step}.csv'
)

final_df.to_csv(os.path.join(pwd_path, df_filename))

plt.plot(xi, yi, c='r')
plt.imshow(img)
plt.show()