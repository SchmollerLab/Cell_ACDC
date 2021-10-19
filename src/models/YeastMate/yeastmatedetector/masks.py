# adapted from detectron2.structures.masks.py (v0.5)
# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import itertools
import numpy as np
from typing import Any, Iterator, List, Union
import torch

from detectron2.structures import Boxes
from detectron2.layers.roi_align import ROIAlign
from detectron2.structures import ROIMasks, PolygonMasks
from detectron2.utils.memory import retry_if_cuda_oom


class BitMasksMulticlass:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,C,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        
        if type(tensor) == torch.Tensor:
            self.image_size = tensor.shape[2:]
        else:
            tensor = torch.as_tensor(tensor, dtype=torch.uint8, device=device)
            self.image_size = tensor.shape[1:]
        
        self.tensor = tensor

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "BitMasksMulticlass":
        return BitMasksMulticlass(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __getitem__(self, item: Union[int, slice, torch.ByteTensor]) -> "BitMasksMulticlass":
        if isinstance(item, int):
            return BitMasksMulticlass(self.tensor[item].view(1, -1))
        m = self.tensor[item]

        return BitMasksMulticlass(m)

    @torch.jit.unused
    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

   
    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        return self.tensor.flatten(1).any(dim=1)

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        """
        Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.
        It has less reconstruction error compared to rasterization with polygons.
        However we observe no difference in accuracy,
        but BitMasks requires more memory to store all the masks.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor:
                A bool tensor of shape (N, mask_size, mask_size), where
                N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        device = self.tensor.device

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

        bit_masks = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)
        output = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
        )

        return output

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(self.tensor, dim=1)
        y_any = torch.any(self.tensor, dim=2)
        for idx in range(self.tensor.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
        return Boxes(boxes)

    @staticmethod
    def cat(bitmasks_list: List["BitMasksMulticlass"]) -> "BitMasksMulticlass":
        """
        Concatenates a list of BitMasks into a single BitMasks

        Arguments:
            bitmasks_list (list[BitMasks])

        Returns:
            BitMasks: the concatenated BitMasks
        """
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, BitMasksMulticlass) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bm.tensor for bm in bitmasks_list], dim=0))
        return cat_bitmasks


class ROIMasksMulticlass(ROIMasks):
    """
    Represent masks by N smaller masks defined in some ROIs. Once ROI boxes are given,
    full-image bitmask can be obtained by "pasting" the mask on the region defined
    by the corresponding ROI box.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor: (N, M, M) mask tensor that defines the mask within each ROI.
        """
        if tensor.dim() != 4:
            raise ValueError("ROIMasksMulticlass must take a masks of 4 dimension.")
        self.tensor = tensor

    def to(self, device: torch.device) -> "ROIMasksMulticlass":
        return ROIMasksMulticlass(self.tensor.to(device))

    @property
    def device(self) -> "device":
        return self.tensor.device

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, item) -> "ROIMasksMulticlass":
        """
        Returns:
            ROIMasksMulticlass: Create a new :class:`ROIMasksMulticlass` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[2:10]`: return a slice of masks.
        2. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        t = self.tensor[item]
        if t.dim() != 4:
            raise ValueError(
                f"Indexing on ROIMasks with {item} returns a tensor with shape {t.shape}!"
            )
        return ROIMasksMulticlass(t)

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    @torch.jit.unused
    def to_bitmasks(self, boxes: torch.Tensor, height, width, threshold=0.5):
        """
        Args:

        """
        from .ops import paste_masks_in_image

        paste = retry_if_cuda_oom(paste_masks_in_image)
        bitmasks = paste(
            self.tensor,
            boxes,
            (height, width),
            threshold=threshold,
        )
        return BitMasksMulticlass(bitmasks)