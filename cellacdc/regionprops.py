import numpy as np
from scipy import ndimage as ndi
import skimage.measure
from . import printl, debugutils
from skimage.measure._regionprops_utils import (
    _normalize_spacing,
)
# WARNING: Developers have already used
#     7 hrs
# to optimize this.
# In addition, implementing these optimizations in the codebase took
#     7 hrs
# Specifically the
#    centroid (huge fain for 3D data)
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
    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        regions.append(
            acdcRegionProperties(
                sl,
                i + 1,
                label_image,
                intensity_image,
                cache,
                spacing=spacing,
                extra_properties=extra_properties,
                offset=offset_arr,
            )
        )

    return regions


# class acdcRegionProperties(_RegionProperties):
#     def __init__(
#         self,
#         slice,
#         label,
#         label_image,
#         intensity_image,
#         cache_active,
#         *,
#         extra_properties=None,
#         spacing=None,
#         offset=None,
#     ):
#         if intensity_image is not None:
#             ndim = label_image.ndim
#             if not (
#                 intensity_image.shape[:ndim] == label_image.shape
#                 and intensity_image.ndim in [ndim, ndim + 1]
#             ):
#                 raise ValueError(
#                     'Label and intensity image shapes must match,'
#                     ' except for channel (last) axis.'
#                 )
#             multichannel = label_image.shape < intensity_image.shape
#         else:
#             multichannel = False

#         self.label = label
#         if offset is None:
#             offset = np.zeros((label_image.ndim,), dtype=int)
#         self._offset = np.array(offset)

#         self._slice = slice
        
#         self._label_image = label_image
#         self._intensity_image = intensity_image

#         self._cache_active = cache_active
#         self._cache = {}
#         self._ndim = label_image.ndim
#         self._multichannel = multichannel
#         self._spatial_axes = tuple(range(self._ndim))
#         if spacing is None:
#             spacing = np.full(self._ndim, 1.0)
#         self._spacing = _normalize_spacing(spacing, self._ndim)
#         self._pixel_area = np.prod(self._spacing)
        
#         self._extra_properties = {}
#         if extra_properties is not None:
#             for func in extra_properties:
#                 name = func.__name__
#                 if hasattr(self, name):
#                     msg = (
#                         f"Extra property '{name}' is shadowed by existing "
#                         f"property and will be inaccessible. Consider "
#                         f"renaming it."
#                     )
#             self._extra_properties = {func.__name__: func for func in extra_properties}
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

    # @property
    # def centroid_weighted(self):
    #     ctr = self.centroid_weighted_local
    #     return tuple(
    #         idx + slc.start * spc
    #         for idx, slc, spc in zip(ctr, self._slice, self._spacing)
    #     )

    # @property
    # @_cached
    # def image_intensity(self):
    #     if self._intensity_image is None:
    #         raise AttributeError('No intensity image specified.')
    #     image = (
    #         self.image
    #         if not self._multichannel
    #         else np.expand_dims(self.image, self._ndim)
    #     )
    #     return self._intensity_image[self._slice] * image

    # @property
    # def coords(self):
    #     indices = np.argwhere(self.image)
    #     object_offset = np.array([self._slice[i].start for i in range(self._ndim)])
    #     return object_offset + indices + self._offset
    
    # @property
    # def coords_scaled(self):
    #     indices = np.argwhere(self.image)
    #     object_offset = np.array([self._slice[i].start for i in range(self._ndim)])
    #     return (object_offset + indices) * self._spacing + self._offset


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
                printl(f"Warning: Object with ID {ID} not found in regionprops.")
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

    def _copy_custom_rp_attributes(self, new_obj, old_obj):
        if old_obj is None:
            return
        new_obj.dead = getattr(old_obj, 'dead', False)
        new_obj.excluded = getattr(old_obj, 'excluded', False)

    def _get_bbox_slices(self, bbox):
        ndim = self.lab.ndim
        if len(bbox) != ndim * 2:
            raise ValueError(
                f'Expected a bounding box with {ndim*2} values, '
                f'got {len(bbox)}.'
            )
        return tuple(
            slice(int(bbox[dim]), int(bbox[dim+ndim])) for dim in range(ndim)
        )

    def _translate_cutout_regionprop(self, obj, offset):
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

        obj._offset = offset_arr.copy()
        obj._cache['slice'] = translated_slice
        obj._cache['bbox'] = translated_bbox
        obj._cache['centroid'] = translated_centroid
        return obj

    def _get_single_obj_regionprop(self, lab, ID):
        mask = lab == ID
        if not np.any(mask):
            return None
        obj = _acdc_regionprops_factory(mask.astype(np.uint8))[0]
        obj.label = ID
        return obj

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
        
    def update_regionprops_via_assignments(
            self, assignments:dict[int, int]
    ):
        """If the lab is completely the same, but only ID changes/swaps have been made

        Parameters
        ----------
        assignments : dict[int, int]
            key: old ID,
            value: new ID
        """
        active_assignments = {
            int(old_ID): int(new_ID)
            for old_ID, new_ID in assignments.items()
            if old_ID in self.IDs_set and old_ID != new_ID
        }
        if not active_assignments:
            return

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
        
    def update_regionprops_via_cutout(
        self, lab, cutout_bbox, specific_IDs=None
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
        printl('Updating rp via cutout...')
        if specific_IDs is not None and not isinstance(specific_IDs, (list, set)):
            specific_IDs = {specific_IDs}
        elif specific_IDs is not None:
            specific_IDs = set(specific_IDs)

        cutout_slices = self._get_bbox_slices(cutout_bbox)
        new_cutout = lab[cutout_slices]
        old_cutout_IDs = self._get_old_cutout_IDs_from_rp(cutout_bbox)
        rp_cutout_new = _acdc_regionprops_factory(new_cutout)
        new_cutout_IDs = set(obj.label for obj in rp_cutout_new)
        new_cutout_IDs.discard(0)
        deleted_IDs = old_cutout_IDs.difference(new_cutout_IDs)
        added_IDs = new_cutout_IDs.difference(old_cutout_IDs)
        preserved_IDs = old_cutout_IDs.intersection(new_cutout_IDs)
        IDs_to_add = (
            added_IDs if specific_IDs is None
            else added_IDs.intersection(specific_IDs)
        )

        if not old_cutout_IDs and not new_cutout_IDs:
            self.lab = lab
            return

        conflicting_IDs = IDs_to_add.intersection(
            self.IDs_set.difference(old_cutout_IDs)
        )
        if conflicting_IDs:
            raise ValueError(
                'Cutout update would reuse IDs that already belong to objects '
                'outside the cutout. Use a full regionprops recomputation.'
            )

        old_rp_by_id = {obj.label: obj for obj in self._rp}
        unaffected_rp = [obj for obj in self._rp if obj.label not in old_cutout_IDs]

        offset = tuple(s.start for s in cutout_slices)
        printl(f"Cutout offset: {offset}")

        new_objs = []
        updated_centroid_IDs = set()
        for obj in rp_cutout_new:
            ID = obj.label
            if ID not in IDs_to_add:
                continue
            if self._is_bbox_touching_cutout_border(obj.bbox, new_cutout.shape):
                # edge case: ID changed is outside the cutout
                new_obj = self._get_single_obj_regionprop(lab, ID)
            else:
                new_obj = self._translate_cutout_regionprop(obj, offset)

            self._copy_custom_rp_attributes(new_obj, old_rp_by_id.get(ID))
            new_objs.append(new_obj)
            updated_centroid_IDs.add(ID)

        for ID in deleted_IDs:
            self._centroid_mapper.pop(ID, None)
            self._centroid_IDs_exact.discard(ID)

        preserved_cutout_rp = [
            old_rp_by_id[ID]
            for ID in preserved_IDs
        ]

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

        self._rp = unaffected_rp + preserved_cutout_rp + new_objs
        self.lab = lab
        self.set_attributes(update_centroid_mapper=False)
            
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
        new_instance.set_attributes(update_centroid_mapper=False)
        return new_instance