import sys
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from timeit import Timer

main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
src_path = os.path.join(main_path, 'src')

sys.path.append(src_path)

from YeaZ.unet import tracking

def np_replace_values(arr, old_values, tracked_values):
    # See method_jdehesa https://stackoverflow.com/questions/45735230/how-to-replace-a-list-of-values-in-a-numpy-array
    old_values = np.asarray(old_values)
    tracked_values = np.asarray(tracked_values)
    n_min, n_max = arr.min(), arr.max()
    replacer = np.arange(n_min, n_max + 1)
    # Mask replacements out of range
    mask = (old_values >= n_min) & (old_values <= n_max)
    replacer[old_values[mask] - n_min] = tracked_values[mask]
    arr = replacer[arr - n_min]
    return arr

def calc_IoA_matrix(lab, prev_lab, rp, prev_rp):
    IDs_prev = []
    IDs_curr_untracked = [obj.label for obj in rp]
    IoA_matrix = np.zeros((len(rp), len(prev_rp)))

    # For each ID in previous frame get IoA with all current IDs
    # Rows: IDs in current frame, columns: IDs in previous frame
    for j, obj_prev in enumerate(prev_rp):
        ID_prev = obj_prev.label
        A_IDprev = obj_prev.area
        IDs_prev.append(ID_prev)
        mask_ID_prev = prev_lab==ID_prev
        intersect_IDs, intersects = np.unique(lab[mask_ID_prev],
                                              return_counts=True)
        for intersect_ID, I in zip(intersect_IDs, intersects):
            if intersect_ID != 0:
                i = IDs_curr_untracked.index(intersect_ID)
                IoA = I/A_IDprev
                IoA_matrix[i, j] = IoA
    return IoA_matrix, IDs_curr_untracked, IDs_prev

def assign(IoA_matrix, IDs_curr_untracked, IDs_prev):
    # Determine max IoA between IDs and assign tracked ID if IoA > 0.4
    max_IoA_col_idx = IoA_matrix.argmax(axis=1)
    unique_col_idx, counts = np.unique(max_IoA_col_idx, return_counts=True)
    counts_dict = dict(zip(unique_col_idx, counts))
    tracked_IDs = []
    old_IDs = []
    for i, j in enumerate(max_IoA_col_idx):
        max_IoU = IoA_matrix[i,j]
        count = counts_dict[j]
        if max_IoU > 0.4:
            tracked_ID = IDs_prev[j]
            if count == 1:
                old_ID = IDs_curr_untracked[i]
            elif count > 1:
                old_ID_idx = IoA_matrix[:,j].argmax()
                old_ID = IDs_curr_untracked[old_ID_idx]
            tracked_IDs.append(tracked_ID)
            old_IDs.append(old_ID)
    return old_IDs, tracked_IDs

def indexAssignment(old_IDs, tracked_IDs, IDs_curr_untracked, lab):
    # Replace untracked IDs with tracked IDs and new IDs with increasing num
    new_untracked_IDs = [ID for ID in IDs_curr_untracked if ID not in old_IDs]
    tracked_lab = lab
    new_tracked_IDs_2 = []
    if new_untracked_IDs:
        # Relabel new untracked IDs with big number to make sure they are unique
        allIDs = IDs_curr_untracked.copy()
        allIDs.extend(tracked_IDs)
        max_ID = max(allIDs)
        new_tracked_IDs = [max_ID*(i+2) for i in range(len(new_untracked_IDs))]
        tracked_lab = np_replace_values(tracked_lab, new_untracked_IDs,
                                        new_tracked_IDs)
        # print('New objects that get a new big ID: ', new_untracked_IDs)
        # print('New big IDs for the new objects: ', new_tracked_IDs)
    if tracked_IDs:
        # Relabel old IDs with respective tracked IDs
        tracked_lab = np_replace_values(tracked_lab, old_IDs, tracked_IDs)
        # print('Old IDs to be tracked: ', old_IDs)
        # print('New IDs replacing old IDs: ', tracked_IDs)
    if new_untracked_IDs:
        # Relabel new untracked IDs sequentially
        max_ID = max(IDs_prev)
        new_tracked_IDs_2 = [max_ID+i+1 for i in range(len(new_untracked_IDs))]
        tracked_lab = np_replace_values(tracked_lab, new_tracked_IDs,
                                             new_tracked_IDs_2)

    return tracked_lab

segm = np.load(r"G:\My Drive\1_MIA_Data\Test_data\Igor\Position_12\Images\exp001_pos2_s12_segm.npz")['arr_0']

lab = segm[-1]*2
rp = regionprops(lab)

print(len(rp))

prev_lab = segm[-2]
prev_rp = regionprops(prev_lab)

n = 1000

IoA_matrix, IDs_curr_untracked, IDs_prev = calc_IoA_matrix(lab, prev_lab, rp, prev_rp)
old_IDs, tracked_IDs = assign(IoA_matrix, IDs_curr_untracked, IDs_prev)

tracked_lab = indexAssignment(old_IDs, tracked_IDs, IDs_curr_untracked, lab)

row_ind, col_ind = linear_sum_assignment(IoA_matrix, maximize=True)

tracked_yeaz = tracking.correspondence(prev_lab, lab)

timer = Timer(lambda: calc_IoA_matrix(lab, prev_lab, rp, prev_rp))
exec_time_IoA = timer.timeit(number=n)

timer = Timer(lambda: assign(IoA_matrix, IDs_curr_untracked, IDs_prev))
exec_time_assignFP = timer.timeit(number=n)

timer = Timer(lambda: indexAssignment(old_IDs, tracked_IDs, IDs_curr_untracked, lab))
exec_time_index = timer.timeit(number=n)

timer = Timer(lambda: linear_sum_assignment(IoA_matrix, maximize=True))
exec_time_assign_scipy = timer.timeit(number=n)

timer = Timer(lambda: tracking.correspondence(prev_lab, lab))
exec_time_yeaz = timer.timeit(number=n)

exec_time_FP = exec_time_IoA+exec_time_assignFP+exec_time_index
exec_time_scipy = exec_time_IoA+exec_time_assign_scipy+exec_time_index

print(f'Total FP exec time = {exec_time_FP*1000/n:.3f} ms')
print(f'Total scipy exec time = {exec_time_scipy*1000/n:.3f} ms')
print(f'Total yeaz exec time = {exec_time_yeaz*1000/n:.3f} ms')

fig, ax = plt.subplots(2,3)

ax[0,0].imshow(prev_lab)
ax[0,1].imshow(lab)
ax[0,2].imshow(tracked_lab)
plt.show()
