import numpy as np
from scipy import ndimage as ndi
from scipy import stats as scipy_stats
import skimage.measure
import cv2
from . import printl, debugutils
from skimage.measure._regionprops_utils import (
    _normalize_spacing,
)
import traceback as traceback

try:
    from cellacdc.precompiled.precompiled_functions import (
        find_all_objects_2D,
        find_all_objects_3D,
    )
    _CYTHON_FIND_OBJECTS = True
except Exception:
    _CYTHON_FIND_OBJECTS = False

try:
    from cellacdc.precompiled.precompiled_functions import most_common_projection_3D
    _CYTHON_MOST_COMMON_PROJECTION = True
except Exception:
    _CYTHON_MOST_COMMON_PROJECTION = False
# WARNING: Developers have already used
#     14 hrs
# to optimize this.
# In addition, implementing these optimizations in the codebase took
#     9 hrs
# Specifically the
#    centroid (huge gain for 3D data)
#    contour caching
#    better find objects implementation to avoid iterating over None lists
#    bbox caching
#    targeted updates to RP
# stuff was targeted. 
# If you decide to try and optimize it further, please update this warning :)

_RegionProperties = skimage.measure._regionprops.RegionProperties
_cached = skimage.measure._regionprops._cached

def _acdc_regionprops_factory(
        label_image,
        intensity_image=None,
        cache=True,
        *,
        extra_properties=None,
        spacing=None,
        offset=None,
    ):
    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        if np.issubdtype(label_image.dtype, bool):
            raise TypeError(
                'Non-integer image types are ambiguous: '
                'use skimage.measure.label to label the connected '
                'components of label_image, '
                'or label_image.astype(np.uint8) to interpret '
                'the True values as a single label.'
            )
        raise TypeError('Non-integer label_image types are ambiguous')

    if offset is None:
        offset_arr = np.zeros((label_image.ndim,), dtype=int)
    else:
        offset_arr = np.asarray(offset)
        if offset_arr.ndim != 1 or offset_arr.size != label_image.ndim:
            raise ValueError(
                'Offset should be an array-like of integers '
                'of shape (label_image.ndim,); '
                f'{offset} was provided.'
            )

    regions = []
    if _CYTHON_FIND_OBJECTS:
        img_uint32 = label_image.astype(np.uint32, copy=False)
        if label_image.ndim == 2:
            out = find_all_objects_2D(img_uint32)
            labels, bboxes = out
            for i in range(len(labels)):
                sl = (slice(int(bboxes[i, 0]), int(bboxes[i, 1])),
                      slice(int(bboxes[i, 2]), int(bboxes[i, 3])))
                regions.append(acdcRegionProperties(
                    sl, int(labels[i]), label_image, intensity_image, cache,
                    spacing=spacing, extra_properties=extra_properties,
                    offset=offset_arr,
                ))
        else:
            out = find_all_objects_3D(img_uint32)
            labels, bboxes = out
            for i in range(len(labels)):
                sl = (slice(int(bboxes[i, 0]), int(bboxes[i, 1])),
                      slice(int(bboxes[i, 2]), int(bboxes[i, 3])),
                      slice(int(bboxes[i, 4]), int(bboxes[i, 5])))
                regions.append(acdcRegionProperties(
                    sl, int(labels[i]), label_image, intensity_image, cache,
                    spacing=spacing, extra_properties=extra_properties,
                    offset=offset_arr,
                ))
    else:
        objects = ndi.find_objects(label_image)
        for i, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            regions.append(acdcRegionProperties(
                sl, i, label_image, intensity_image, cache,
                spacing=spacing, extra_properties=extra_properties,
                offset=offset_arr,
            ))
    return regions

class acdcRegionProperties(_RegionProperties):
    def __init__(
        self,
        slice,
        label,
        label_image,
        intensity_image,
        cache_active,
        *,
        extra_properties=None,
        spacing=None,
        offset=None,
    ):
        super().__init__(
            slice, label, label_image, intensity_image, cache_active,
            extra_properties=extra_properties, spacing=spacing, offset=offset
        )    
    # @property
    # @_cached
    # def slice(self):
    #     # scale slice with offset
    #     return tuple(
    #         slice(self._slice[i].start + self._offset[i],
    #               self._slice[i].stop + self._offset[i])
    #         for i in range(self._ndim)
    #     )
    
    @property
    def image(self):
        """Return cached object mask from the current label image."""
        imgage = self._cache.get('image')
        if imgage is None or not np.any(imgage):
            self._cache['image'] = self._label_image[self._slice] == self.label
        
        return self._cache['image']

    @property
    @_cached
    def bbox(self):
        """
        Returns
        -------
        A tuple of the bounding box's start coordinates for each dimension,
        followed by the end coordinates for each dimension.
        """
        return tuple(
            [self.slice[i].start for i in range(self._ndim)]
            + [self.slice[i].stop for i in range(self._ndim)]
        )
    
    @property
    @_cached # slow for 3D data, better cache it
    def centroid(self):
        return super().centroid

    @property
    @_cached
    def contour(self):
        contours = self._contours_local(retrieve_mode=cv2.RETR_EXTERNAL)
        if not contours:
            return np.empty((0, 2), dtype=np.int32)

        contour = max(contours, key=len)
        contour = np.squeeze(contour, axis=1)
        contour = np.vstack((contour, contour[0]))
        return contour + self._xy_offset

    @property
    @_cached
    def contour_all(self):
        # Include both outer boundaries and holes.
        contours = self._contours_local(retrieve_mode=cv2.RETR_CCOMP)
        if not contours:
            return []
        offset = self._xy_offset
        return [np.squeeze(cont, axis=1) + offset for cont in contours]

    @property
    @_cached
    def _xy_offset(self):
        if self._ndim != 2:
            raise AttributeError('contour is only supported for 2D objects.')
        slc = self.slice
        return np.array([slc[1].start, slc[0].start], dtype=np.int32)

    def _contours_local(self, retrieve_mode=cv2.RETR_EXTERNAL):
        if self._ndim != 2:
            raise AttributeError('contour is only supported for 2D objects.')
        obj_image = np.ascontiguousarray(self.image, dtype=np.uint8)
        contours, _ = cv2.findContours(
            obj_image, retrieve_mode, cv2.CHAIN_APPROX_NONE
        )
        return contours

class acdcRegionprops:
    def __init__(
            self,
            lab,
            acdc_df=None,
            centroids_loaded=None,
            IDs_loaded=None,
            centroids_IDs_exact_loaded=None,
            ID_to_idx_loaded=None,
            precache_centroids=True,
            **kwargs,
        ):
        self.lab = lab
        self.acdc_df = acdc_df
        self._rp = _acdc_regionprops_factory(lab, **kwargs)
        self.is3D = self.lab.ndim == 3
        self._slice_rps = {
            'z': {},
            'y': {},
            'x': {},
        }
        self._proj_rps = {
            'z': {},
            'y': {},
            'x': {},
        }
        self._proj_labs = {
            'z': {},
            'y': {},
            'x': {},
        }
        self._centroid_mapper = {}
        self._centroid_IDs_exact = set()
        if IDs_loaded is None or ID_to_idx_loaded is None:
            self.set_attributes(update_centroid_mapper=False)
        else:
            self.ID_to_idx = ID_to_idx_loaded
            self.IDs_set = set(IDs_loaded)
            self.IDs = list(self.IDs_set)

        if centroids_IDs_exact_loaded is not None and centroids_loaded is not None:
            self._centroid_mapper = centroids_loaded
            self._centroid_IDs_exact = set(centroids_IDs_exact_loaded)
        elif precache_centroids:
            self.precache_centroids()
            
        else:
            self._centroid_mapper = dict()

    def get_slice_rp(self, slice_number, slicing='z'):
        if not self.is3D:
            raise ValueError('Slice-specific regionprops are only supported for 3D labels.')

        slicing = self._normalize_slicing(slicing)
        slice_number = int(slice_number)
        self._validate_slice_number(slice_number, slicing)

        rp = self._slice_rps[slicing].get(slice_number)
        if rp is None:
            lab_slice = self._get_lab_slice(self.lab, slice_number, slicing)
            rp = acdcRegionprops(lab_slice, precache_centroids=False)
            self._slice_rps[slicing][slice_number] = rp
        return rp

    def get_proj_rp(self, kind='max', slicing='z'):
        if not self.is3D:
            raise ValueError('Projection-specific regionprops are only supported for 3D labels.')

        slicing = self._normalize_slicing(slicing)
        kind = self._normalize_projection_kind(kind)

        rp = self._proj_rps[slicing].get(kind)
        if rp is None:
            lab_proj = self._proj_labs[slicing].get(kind)
            if lab_proj is None:
                lab_proj = self._get_lab_projection(self.lab, slicing=slicing, kind=kind)
                self._proj_labs[slicing][kind] = lab_proj
            rp = acdcRegionprops(lab_proj, precache_centroids=False)
            self._proj_rps[slicing][kind] = rp
        return rp

    def get_obj_from_slice_rp(self, ID, slice_number, slicing='z', warn=True):
        rp = self.get_slice_rp(slice_number, slicing=slicing)
        return rp.get_obj_from_ID(ID, warn=warn)

    def get_obj_from_proj_rp(self, ID, kind='max', slicing='z', warn=True):
        kind = self._normalize_projection_kind(kind)
        rp = self.get_proj_rp(kind=kind, slicing=slicing)
        return rp.get_obj_from_ID(ID, warn=warn)

    def __iter__(self):
        return iter(self._rp)

    def __len__(self):
        return len(self._rp)

    def __getitem__(self, idx):
        return self._rp[idx]

    def __setitem__(self, idx, value):
        self._rp[idx] = value

    def __repr__(self):
        return repr(self._rp)

    def _normalize_slicing(self, slicing):
        slicing = str(slicing).lower()
        if slicing not in ('z', 'y', 'x'):
            raise ValueError(
                f'Invalid slicing "{slicing}". Valid options are "z", "y", and "x".'
            )
        return slicing

    def _slice_axis_index(self, slicing):
        axis_map = {'z': 0, 'y': 1, 'x': 2}
        return axis_map[slicing]

    def _normalize_projection_kind(self, kind):
        kind = str(kind).lower().strip()
        kind_norm = kind.replace('-', ' ').replace('_', ' ')

        if kind_norm.startswith('max'):
            return 'max'
        if kind_norm.startswith('mean'):
            return 'mean'
        if kind_norm.startswith('median'):
            return 'median'
        if kind_norm.startswith('most common') or kind_norm.startswith('mode'):
            return 'most_common'

        if kind not in ('max', 'mean', 'median', 'most_common'):
            raise ValueError(
                f'Invalid projection kind "{kind}". '
                'Valid options are "max", "mean", "median", and "most_common".'
            )
        return kind

    def _validate_slice_number(self, slice_number, slicing):
        axis = self._slice_axis_index(slicing)
        axis_size = self.lab.shape[axis]
        if slice_number < 0 or slice_number >= axis_size:
            raise IndexError(
                f'Slice number {slice_number} is out of bounds for slicing "{slicing}" '
                f'with size {axis_size}.'
            )

    def _has_initialized_slice_rps(self):
        return any(len(slice_dict) > 0 for slice_dict in self._slice_rps.values())

    def _has_initialized_proj_rps(self):
        return any(len(proj_dict) > 0 for proj_dict in self._proj_rps.values())

    def _iter_initialized_slice_rps(self):
        for slicing, slice_dict in self._slice_rps.items():
            for slice_number, rp in slice_dict.items():
                yield slicing, slice_number, rp

    def _iter_initialized_proj_rps(self):
        for slicing, proj_dict in self._proj_rps.items():
            for kind, rp in proj_dict.items():
                yield slicing, kind, rp

    def _get_lab_slice(self, lab, slice_number, slicing):
        if lab.ndim != 3:
            raise ValueError(
                f'Slice-specific regionprops are only supported for 3D labels, got {lab.ndim}D.'
            )

        slicing = self._normalize_slicing(slicing)
        if slicing == 'z':
            return lab[slice_number, :, :]
        if slicing == 'y':
            return lab[:, slice_number, :]
        return lab[:, :, slice_number]

    def _get_lab_projection(self, lab, slicing='z', kind='max'):
        if lab.ndim != 3:
            raise ValueError(
                f'Projection-specific regionprops are only supported for 3D labels, got {lab.ndim}D.'
            )

        axis = self._slice_axis_index(self._normalize_slicing(slicing))
        kind = self._normalize_projection_kind(kind)
        if kind == 'max':
            return np.max(lab, axis=axis)

        if kind == 'most_common':
            return self._compute_most_common_projection(lab, axis=axis)

        if kind == 'mean':
            projected = np.mean(lab, axis=axis)
        else:
            projected = np.median(lab, axis=axis)

        # Regionprops requires integer labels.
        return np.rint(projected).astype(lab.dtype, copy=False)

    def _compute_most_common_projection(self, lab, axis):
        if _CYTHON_MOST_COMMON_PROJECTION:
            lab_uint32 = lab.astype(np.uint32, copy=False)
            projected = most_common_projection_3D(lab_uint32, int(axis))
            return projected.astype(lab.dtype, copy=False)

        moved = np.moveaxis(lab, axis, 0)
        projected = scipy_stats.mode(moved, axis=0, keepdims=False).mode
        return projected.astype(lab.dtype, copy=False)

    def _get_projection_patch_slices(self, slicing, cutout_bbox):
        z0, y0, x0, z1, y1, x1 = [int(v) for v in cutout_bbox]
        if slicing == 'z':
            return (slice(y0, y1), slice(x0, x1))
        if slicing == 'y':
            return (slice(z0, z1), slice(x0, x1))
        return (slice(z0, z1), slice(y0, y1))

    def _compute_most_common_projection_patch(self, slicing, cutout_bbox):
        z0, y0, x0, z1, y1, x1 = [int(v) for v in cutout_bbox]
        if slicing == 'z':
            patch_lab = self.lab[:, y0:y1, x0:x1]
        elif slicing == 'y':
            patch_lab = self.lab[z0:z1, :, x0:x1]
        else:
            patch_lab = self.lab[z0:z1, y0:y1, :]

        axis = self._slice_axis_index(slicing)
        return self._compute_most_common_projection(patch_lab, axis=axis)

    def _update_cached_most_common_projection_locally(self, slicing, cutout_bbox):
        lab_proj = self._get_cached_or_new_lab_projection(slicing, 'most_common')
        patch_slices = self._get_projection_patch_slices(slicing, cutout_bbox)
        if any(slc.start >= slc.stop for slc in patch_slices):
            return lab_proj

        patch = self._compute_most_common_projection_patch(slicing, cutout_bbox)
        lab_proj[patch_slices] = patch
        self._proj_labs[slicing]['most_common'] = lab_proj
        return lab_proj

    def _get_cached_or_new_lab_projection(self, slicing, kind):
        lab_proj = self._proj_labs[slicing].get(kind)
        if lab_proj is None:
            lab_proj = self._get_lab_projection(self.lab, slicing=slicing, kind=kind)
            self._proj_labs[slicing][kind] = lab_proj
        return lab_proj

    def _replace_cached_lab_projection(self, slicing, kind):
        lab_proj = self._get_lab_projection(self.lab, slicing=slicing, kind=kind)
        self._proj_labs[slicing][kind] = lab_proj
        return lab_proj

    def _apply_assignments_to_lab_from_rp(self, rp, lab, assignments):
        active_assignments = {
            int(old_ID): int(new_ID)
            for old_ID, new_ID in assignments.items()
            if old_ID != new_ID
        }
        if not active_assignments:
            return lab

        dst = lab.copy()
        for obj in rp:
            new_ID = active_assignments.get(obj.label)
            if new_ID is None:
                continue
            dst[obj.slice][obj.image] = new_ID
        return dst

    def _apply_deletions_to_lab_from_rp(self, rp, lab, IDs_to_delete):
        IDs_to_delete = set(IDs_to_delete)
        if not IDs_to_delete:
            return lab

        dst = lab.copy()
        for obj in rp:
            if obj.label in IDs_to_delete:
                dst[obj.slice][obj.image] = 0
        return dst

    def _sync_initialized_slice_rps_via_assignments(self, assignments):
        if not self._has_initialized_slice_rps():
            return

        for slicing, slice_number, rp in self._iter_initialized_slice_rps():
            lab_slice = self._get_lab_slice(self.lab, slice_number, slicing)
            rp.update_regionprops_via_assignments(assignments, lab_slice)

    def _sync_initialized_proj_rps_via_assignments(self, assignments):
        if not self._has_initialized_proj_rps():
            return

        for slicing, kind, rp in self._iter_initialized_proj_rps():
            lab_proj = self._get_cached_or_new_lab_projection(slicing, kind)
            lab_proj = self._apply_assignments_to_lab_from_rp(
                rp, lab_proj, assignments
            )
            self._proj_labs[slicing][kind] = lab_proj
            rp.update_regionprops_via_assignments(assignments, lab_proj)

    def _sync_initialized_slice_rps_via_deletions(self, IDs_to_delete):
        if not self._has_initialized_slice_rps():
            return

        for _, _, rp in self._iter_initialized_slice_rps():
            rp.update_regionprops_via_deletions(IDs_to_delete)

    def _sync_initialized_proj_rps_via_deletions(self, IDs_to_delete):
        if not self._has_initialized_proj_rps():
            return

        for slicing, kind, rp in self._iter_initialized_proj_rps():
            lab_proj = self._get_cached_or_new_lab_projection(slicing, kind)
            lab_proj = self._apply_deletions_to_lab_from_rp(
                rp, lab_proj, IDs_to_delete
            )
            self._proj_labs[slicing][kind] = lab_proj
            rp.update_regionprops_via_deletions(IDs_to_delete)

    def _sync_initialized_slice_rps_via_update(self, specific_IDs_update_centroids=None):
        if not self._has_initialized_slice_rps():
            return

        for slicing, slice_number, rp in self._iter_initialized_slice_rps():
            lab_slice = self._get_lab_slice(self.lab, slice_number, slicing)
            rp.update_regionprops(
                lab_slice,
                specific_IDs_update_centroids=specific_IDs_update_centroids,
            )

    def _sync_initialized_proj_rps_via_update(
            self,
            specific_IDs_update_centroids=None,
            cutout_bbox=None,
        ):
        if not self._has_initialized_proj_rps():
            return

        for slicing, kind, rp in self._iter_initialized_proj_rps():
            if cutout_bbox is not None and kind == 'most_common':
                lab_proj = self._update_cached_most_common_projection_locally(
                    slicing, cutout_bbox
                )
            else:
                lab_proj = self._replace_cached_lab_projection(slicing, kind)
            rp.update_regionprops(
                lab_proj,
                specific_IDs_update_centroids=specific_IDs_update_centroids,
            )

    def _normalize_cutout_bbox(self, cutout_bbox):
        """Normalize cutout_bbox to always be 3D (6 values) for 3D data.
        
        Automatically expands 2D bbox (4 values: y_start, x_start, y_end, x_end)
        to 3D bbox (6 values: z_start, y_start, x_start, z_end, y_end, x_end) 
        covering all z-slices.
        """
        if self.is3D:
            if len(cutout_bbox) == 4:
                # 2D bbox: expand to 3D with full z range
                y_start, x_start, y_end, x_end = cutout_bbox
                return (0, y_start, x_start, self.lab.shape[0], y_end, x_end)
            elif len(cutout_bbox) != 6:
                raise ValueError(
                    'For 3D labels, cutout_bbox should have 4 values (2D) or 6 values (3D): '
                    f'got {len(cutout_bbox)}.'
                )
        else:
            if len(cutout_bbox) != 4:
                raise ValueError(
                    'For 2D labels, cutout_bbox should have 4 values (y_start, x_start, y_end, x_end), '
                    f'got {len(cutout_bbox)}.'
                )
        return cutout_bbox

    def _get_centroid_df_from_df(self):
        if self.acdc_df is None or len(self.acdc_df) == 0:
            return {}

        centroid_cols = ['y_centroid', 'x_centroid']
        if self.is3D and 'z_centroid' in self.acdc_df.columns:
            centroid_cols = ['z_centroid', 'y_centroid', 'x_centroid']

        if not set(centroid_cols).issubset(self.acdc_df.columns):
            return {}

        if 'Cell_ID' in self.acdc_df.columns:
            centroid_df = self.acdc_df.set_index('Cell_ID')[centroid_cols]
        elif 'ID' in self.acdc_df.columns:
            centroid_df = self.acdc_df.set_index('ID')[centroid_cols]
        else:
            centroid_df = self.acdc_df[centroid_cols]

        return {
            int(ID): tuple(values)
            for ID, values in centroid_df.iterrows()
        }

    def _get_bbox_centers_mapper(
            self, objs=None, IDs_to_include=None, IDs_to_exclude=None
        ):
        if objs is None and not self._rp:
            return {}

        if objs is None:
            if IDs_to_include is None:
                IDs_to_include = (
                    self.IDs_set.difference(IDs_to_exclude)
                    if IDs_to_exclude is not None else self.IDs_set
                )
            ids = set(IDs_to_include)
            objs = [obj for obj in self._rp if obj.label in ids]

        if not objs:
            return {}

        ndim = 2 if not self.is3D else 3
        labels = np.empty(len(objs), dtype=int)
        bboxes = np.empty((len(objs), ndim * 2), dtype=float)
        for i, obj in enumerate(objs):
            labels[i] = obj.label
            bboxes[i] = obj.bbox

        centers = (bboxes[:, :ndim] + bboxes[:, ndim:]) / 2.0
        return {
            int(label): tuple(center)
            for label, center in zip(labels, centers)
        }

    def precache_centroids(self):
        centroid_df = self._get_centroid_df_from_df()
        IDs_from_df = set(centroid_df)
        IDs_missing_centroid = self.IDs_set.difference(IDs_from_df)
        bbox_centers_mapper = self._get_bbox_centers_mapper(
            IDs_to_include=IDs_missing_centroid
        )
        self._centroid_mapper = {**bbox_centers_mapper, **centroid_df}
        self._centroid_IDs_exact = IDs_from_df
        
    def set_attributes(self, deleted_IDs=None, update_centroid_mapper=True):
        self.ID_to_idx = {obj.label: idx for idx, obj in enumerate(self._rp)}
        # Update IDs and IDs_set separately and explicitly
        self.IDs_set = set(self.ID_to_idx)
        self.IDs = list(self.IDs_set)

        if not update_centroid_mapper:
            return
        if deleted_IDs is not None:
            for ID in deleted_IDs:
                self._centroid_mapper.pop(ID, None)
                self._centroid_IDs_exact.discard(ID)
        else:
            self._centroid_mapper = {
                ID: centroid
                for ID, centroid in self._centroid_mapper.items()
                if ID in self.IDs_set
            }
            self._centroid_IDs_exact.intersection_update(self.IDs_set)

    def get_obj_from_ID(self, ID, warn=True):
        idx = self.ID_to_idx.get(ID, None)
        if idx is not None:
            return self._rp[idx]
        else:
            if warn:
                # get caller info
                debugutils.print_call_stack()
                print(f"Warning: Object with ID {ID} not found in regionprops.")
            return None
        
    def delete_IDs(self, IDs_to_delete: set[int], update_other_attrs=True):
        if not IDs_to_delete:
            return

        self._rp = [
            obj for obj in self._rp if obj.label not in IDs_to_delete
        ]
        
        if not update_other_attrs:
            return
        self.set_attributes(deleted_IDs=IDs_to_delete)    

    def _get_IDs_to_update_centroids(
            self, lab, objs, specific_IDs_update_centroids=None
        ):
        if specific_IDs_update_centroids is not None:
            return set(specific_IDs_update_centroids)

        obj_to_update = set()
        for obj in objs:
            has_to_update = False
            ID = obj.label
            old_centroid = self._centroid_mapper.get(ID, None)
            if old_centroid is not None:
                rounded_centroid = tuple(np.round(old_centroid).astype(int))
                try:
                    ID_lab = lab[rounded_centroid]
                except Exception:
                    has_to_update = True
                else:
                    if ID_lab != ID:
                        has_to_update = True
            else:
                has_to_update = True

            if has_to_update:
                obj_to_update.add(ID)

        return obj_to_update
    
    def update_regionprops(
            self, lab, specific_IDs_update_centroids=None,
            update_centroids=True
        ):
        old_rp_by_id = {obj.label: obj for obj in self._rp}
        
        new_rp = _acdc_regionprops_factory(lab)
        
        if update_centroids:
            # Verify that the cached centroid is still inside the object mask.
            obj_to_update = self._get_IDs_to_update_centroids(
                lab, new_rp,
                specific_IDs_update_centroids=specific_IDs_update_centroids
            )
            
            bbox_centers_mapper = self._get_bbox_centers_mapper(
                objs=[obj for obj in new_rp if obj.label in obj_to_update]
            )

            # update centroids
            self._centroid_mapper.update(bbox_centers_mapper)
            
            # remove from exact set if we updated the centroid
            self._centroid_IDs_exact.difference_update(obj_to_update)

        for obj in new_rp:
            self._copy_custom_rp_attributes(obj, old_rp_by_id.get(obj.label))

        self._rp = new_rp
        self.lab = lab
        self.set_attributes()
        self._sync_initialized_slice_rps_via_update(
            specific_IDs_update_centroids=specific_IDs_update_centroids
        )
        self._sync_initialized_proj_rps_via_update(
            specific_IDs_update_centroids=specific_IDs_update_centroids
        )

    def _copy_custom_rp_attributes(self, new_obj, old_obj):
        if old_obj is None:
            return
        new_obj.dead = getattr(old_obj, 'dead', False)
        new_obj.excluded = getattr(old_obj, 'excluded', False)

    def _get_bbox_slices(self, bbox, depth_axis=None):
        ndim = self.lab.ndim
        if len(bbox) != ndim * 2:
            raise ValueError(
                f'Expected a bounding box with {ndim*2} values, '
                f'got {len(bbox)}.'
            )
        
        return tuple(
            slice(int(bbox[dim]), int(bbox[dim+ndim])) for dim in range(ndim)
        )

    def _translate_cutout_regionprop(self, obj, offset, lab):
        offset_arr = np.asarray(offset)
        centroid = obj.centroid
        translated_slice = tuple(
            slice(
                obj._slice[dim].start + offset_arr[dim],
                obj._slice[dim].stop + offset_arr[dim],
            )
            for dim in range(obj._ndim)
        )
        translated_bbox = tuple(
            [slc.start for slc in translated_slice]
            + [slc.stop for slc in translated_slice]
        )
        translated_centroid = tuple(
            coord + offset_arr[dim]
            for dim, coord in enumerate(centroid)
        )

        obj._label_image = lab
        obj._slice = translated_slice
        obj.slice = translated_slice
        obj._offset = np.zeros_like(offset_arr)
        obj._cache['slice'] = translated_slice
        obj._cache['bbox'] = translated_bbox
        obj._cache['centroid'] = translated_centroid
        return obj

    def _get_separate_obj_regionprops(self, lab, IDs):
        IDs = tuple(int(ID) for ID in IDs)
        if not IDs:
            return {}

        mask = np.isin(lab, IDs)
        if not np.any(mask):
            return {}

        isolated_lab = np.zeros_like(lab)
        isolated_lab[mask] = lab[mask]
        return {
            obj.label: obj
            for obj in _acdc_regionprops_factory(isolated_lab)
            if obj.label in IDs
        }

    def _is_bbox_touching_cutout_border(self, bbox, shape):
        ndim = len(shape)
        for dim in range(ndim):
            if bbox[dim] == 0 or bbox[dim+ndim] == shape[dim]:
                return True
        return False

    def _obj_intersects_bbox(self, obj, bbox):
        ndim = self.lab.ndim
        obj_bbox = obj.bbox
        for dim in range(ndim):
            start = max(int(obj_bbox[dim]), int(bbox[dim]))
            stop = min(int(obj_bbox[dim+ndim]), int(bbox[dim+ndim]))
            if start >= stop:
                return False

        return True

    def _get_old_cutout_IDs_from_rp(self, cutout_bbox):
        return {
            obj.label for obj in self._rp
            if self._obj_intersects_bbox(obj, cutout_bbox)
        }

    def _set_label_image(self, lab, objs=None, clear_cache=False):
        if lab is None:
            return

        self.lab = lab
        if objs is None:
            objs = self._rp

        for obj in objs:
            obj._label_image = lab
            if clear_cache:
                obj._cache.clear()
        
    def update_regionprops_via_assignments(
            self, assignments:dict[int, int], lab
    ):
        """If the lab is completely the same, but only ID changes/swaps have been made

        Parameters
        ----------
        assignments : dict[int, int]
            key: old ID,
            value: new ID
        lab : np.ndarray, optional
            Updated label image. When provided, regionprops objects are rebound
            to this image so properties such as ``image`` stay consistent after
            the ID remap.
        """
        active_assignments = {
            int(old_ID): int(new_ID)
            for old_ID, new_ID in assignments.items()
            if old_ID in self.IDs_set and old_ID != new_ID
        }
        if not active_assignments:
            self._set_label_image(lab)
            self._sync_initialized_slice_rps_via_assignments({})
            self._sync_initialized_proj_rps_via_assignments({})
            return

        # if not active_assignments:
        #     if lab is not None:
        #         self._set_label_image(lab)
        #     return

        # remapped_IDs = set()
        # for obj in self._rp:
        #     old_ID = obj.label
        #     new_ID = active_assignments.get(old_ID, old_ID)
        #     if new_ID in remapped_IDs:
        #         raise ValueError(
        #             'Assignments would create duplicate IDs in regionprops. '
        #             'Use a full regionprops recomputation for merges.'
        #         )
        #     remapped_IDs.add(new_ID)

        centroid_mapper = {
            active_assignments.get(ID, ID): centroid
            for ID, centroid in self._centroid_mapper.items()
            # if active_assignments.get(ID, ID) in remapped_IDs
        }
        centroid_IDs_exact = {
            active_assignments.get(ID, ID)
            for ID in self._centroid_IDs_exact
            # if active_assignments.get(ID, ID) in remapped_IDs
        }

        # Rebind first so any property access during remap sees the current lab.
        self._set_label_image(lab)

        for obj in self._rp:
            old_ID = obj.label
            new_ID = active_assignments.get(old_ID, old_ID)
            obj.label = new_ID
            # if obj.area == 0:
            #     # if area is 0, centroid is not defined and we should not trust the cached one
            #     print("area 0...")

        self._centroid_mapper = centroid_mapper
        self._centroid_IDs_exact = centroid_IDs_exact
        self.set_attributes(update_centroid_mapper=False) # update the mapper
        self._sync_initialized_slice_rps_via_assignments(active_assignments)
        self._sync_initialized_proj_rps_via_assignments(active_assignments)

    def update_regionprops_via_deletions(
            self, IDs_to_delete: set[int]
    ):
        """If the lab is completely the same, but only some IDs have been deleted

        Parameters
        ----------
        IDs_to_delete : set[int]
            IDs to delete
        """
        IDs_to_delete = set(IDs_to_delete).intersection(self.IDs_set)
        if not IDs_to_delete:
            return
        self._rp = [obj for obj in self._rp if obj.label not in IDs_to_delete]
        self.set_attributes(deleted_IDs=IDs_to_delete) # for updating the IDs to indx, centroid mapper
        self._sync_initialized_slice_rps_via_deletions(IDs_to_delete)
        self._sync_initialized_proj_rps_via_deletions(IDs_to_delete)
        
    def update_regionprops_via_cutout(
        self, lab, cutout_bbox, specific_IDs=None, depth_axis=None
    ):
        """Only relabels the regionprops of a specific cutout.
        Is only faster for small cutouts. I dont have a number, but I would say
        less than 30% of total image size.

        Parameters
        ----------
        cutout_lab : np.ndarray
            The labeled cutout image.
        cutout_bbox : tuple[int, int, int, int]
            The bounding box of the cutout in the format (min_row, min_col, max_row, max_col).
        """
        if specific_IDs is not None and not isinstance(specific_IDs, (list, set, np.ndarray, tuple)):
            specific_IDs = {specific_IDs}
        elif specific_IDs is not None:
            specific_IDs = set(specific_IDs)

        self.lab = lab
        # Normalize bbox to 3D (expands 2D bbox to full z-range)
        cutout_bbox = self._normalize_cutout_bbox(cutout_bbox)
        cutout_slices = self._get_bbox_slices(cutout_bbox, depth_axis=depth_axis)
        new_cutout = lab[cutout_slices]
        old_cutout_IDs = self._get_old_cutout_IDs_from_rp(cutout_bbox)
        rp_cutout_new = _acdc_regionprops_factory(new_cutout)
        new_cutout_IDs = set(obj.label for obj in rp_cutout_new)

        if not old_cutout_IDs and not new_cutout_IDs:
            return
        
        target_IDs = (
            old_cutout_IDs.union(new_cutout_IDs)
            if specific_IDs is None
            else old_cutout_IDs.union(new_cutout_IDs).intersection(specific_IDs)
        )
        
        deleted_target_IDs = old_cutout_IDs.difference(new_cutout_IDs).intersection(
            target_IDs
        )
        
        refreshed_IDs = new_cutout_IDs.intersection(target_IDs)

        conflicting_IDs = refreshed_IDs.difference(old_cutout_IDs).intersection(
            self.IDs_set.difference(old_cutout_IDs)
        )
        if conflicting_IDs:
            raise ValueError(
                'Cutout update would reuse IDs that already belong to objects '
                'outside the cutout. Use a full regionprops recomputation.'
            )

        old_rp_by_id = {obj.label: obj for obj in self._rp}
        IDs_to_replace = old_cutout_IDs.intersection(target_IDs)
        unaffected_rp = [obj for obj in self._rp if obj.label not in IDs_to_replace]

        offset = tuple(s.start for s in cutout_slices)

        border_touching_IDs = {
            obj.label
            for obj in rp_cutout_new
            if obj.label in refreshed_IDs
            and self._is_bbox_touching_cutout_border(obj.bbox, new_cutout.shape)
        }
        separate_objs = self._get_separate_obj_regionprops(lab, border_touching_IDs)

        new_objs = []
        updated_centroid_IDs = set()
        for obj in rp_cutout_new:
            ID = obj.label
            if ID not in refreshed_IDs:
                continue
            if ID in border_touching_IDs:
                # edge case: ID changed is outside the cutout
                new_obj = separate_objs.get(ID)
                if new_obj is None:
                    continue
            else:
                new_obj = self._translate_cutout_regionprop(obj, offset, lab)

            self._copy_custom_rp_attributes(new_obj, old_rp_by_id.get(ID))
            new_objs.append(new_obj)
            updated_centroid_IDs.add(ID)

        for ID in deleted_target_IDs:
            self._centroid_mapper.pop(ID, None)
            self._centroid_IDs_exact.discard(ID)

        if updated_centroid_IDs:
            obj_to_update = self._get_IDs_to_update_centroids(
                lab, new_objs,
                specific_IDs_update_centroids=updated_centroid_IDs
            )
            
            self._centroid_mapper.update(
                self._get_bbox_centers_mapper(
                    objs=[obj for obj in new_objs if obj.label in obj_to_update]
                )
            )
            self._centroid_IDs_exact.difference_update(obj_to_update)

        self._rp = unaffected_rp + new_objs
        self._set_label_image(lab)
        self.set_attributes(update_centroid_mapper=False)
        self._sync_initialized_slice_rps_via_update(
            specific_IDs_update_centroids=target_IDs
        )
        self._sync_initialized_proj_rps_via_update(
            specific_IDs_update_centroids=target_IDs,
            cutout_bbox=cutout_bbox,
        )
            
    def get_centroid(self, ID, exact=False):
        if exact and ID not in self._centroid_IDs_exact:
            obj = self.get_obj_from_ID(ID)
            centroid = obj.centroid
            try:
                int(centroid[0])
            except (TypeError, ValueError):
                print(f"Warning: Centroid for ID {ID} is not a valid coordinate: {centroid}. "
                      f"Object size: {obj.bbox}. Returning None.")
                return None
            self._centroid_mapper[ID] = centroid
            self._centroid_IDs_exact.add(ID)
            return centroid
        
        centroid = self._centroid_mapper.get(ID, None)
        if centroid is None:
            # add centroid to mapper if not found
            objs = [self.get_obj_from_ID(ID)]
            bbox_centers_mapper = self._get_bbox_centers_mapper(objs=objs)
            self._centroid_mapper.update(bbox_centers_mapper)
            centroid = self._centroid_mapper.get(ID, None)
        return centroid
    
    def copy(self):
        new_instance = acdcRegionprops(
            self.lab, precache_centroids=False
        )
        new_instance._rp = [obj for obj in self._rp]
        new_instance._centroid_mapper = self._centroid_mapper.copy()
        new_instance._centroid_IDs_exact = self._centroid_IDs_exact.copy()
        for slicing, slice_number, rp in self._iter_initialized_slice_rps():
            new_instance._slice_rps[slicing][slice_number] = rp.copy()
        for slicing, kind, rp in self._iter_initialized_proj_rps():
            new_instance._proj_rps[slicing][kind] = rp.copy()
        for slicing, proj_labs in self._proj_labs.items():
            for kind, lab_proj in proj_labs.items():
                new_instance._proj_labs[slicing][kind] = lab_proj.copy()
        new_instance.set_attributes(update_centroid_mapper=False)
        return new_instance