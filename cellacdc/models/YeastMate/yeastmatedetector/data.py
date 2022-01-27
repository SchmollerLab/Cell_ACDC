import os
import copy
import torch
import json
import numpy as np
from glob import glob

from skimage.io import imread
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)
from detectron2.data import DatasetMapper, detection_utils as utils

from .masks import BitMasksMulticlass

class DictGetter:
    def __init__(self, dataset, train_path=None):
        self.dataset = dataset
        self.train_path = train_path

    def get_train_dicts(self):
        if self.train_path:
            return get_multi_masks(self.train_path)

        else:
            raise ValueError("Training data path is not set!")

def annotations_to_instances(annos, image_size, mask_format="bitmaskmulticlass"):
    """
    Adapted from detectron2 (https://github.com/facebookresearch/detectron2)
    """

    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    segms = [obj["segmentation"] for obj in annos]

    if len(annos):
        if mask_format == "bitmaskmulticlass":
            masks = BitMasksMulticlass(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in segms])
            )
            target.gt_masks = masks
        elif mask_format == "bitmask":
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in segms])
            )
            target.gt_masks = masks

    return target

def get_multi_masks(root_dir):
    imglist = glob(os.path.join(root_dir, '*.tif'))

    imglist = [path for path in imglist if 'mask' not in path]
    imglist.sort()

    dataset_dicts = []
    for idx, filename in enumerate(imglist):
        record = {}
        objs = []

        shapecheck = imread(filename).shape

        if len(shapecheck) == 2:
            height, width = shapecheck
        elif len(shapecheck) == 3:
            height, width = shapecheck[0:2]
        elif len(shapecheck) == 4:
            height, width = shapecheck[1:3]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        name = filename.replace('.tif', '_mask.tif')

        mask = imread(name).astype(np.uint16)
        with open(name.replace('_mask.tif', '_detections.json'), 'r') as file:
            gt_dict = json.load(file)

        rp_dict = {}
        rp_objs = regionprops(mask)
        for rp in rp_objs:
            rp_dict[rp.label] = rp

        fullmask = np.zeros((mask.shape[0], mask.shape[1], 7), dtype=np.uint16)
        fullmask[:,:,0] = mask

        for key, thing in gt_dict['detections'].items():
            try:
                subclass_idx = int(thing['class'][0].split('.')[1])

                if subclass_idx > 0: continue

                class_idx = int(thing['class'][0].split('.')[0])

            except IndexError:
                class_idx = int(thing['class'][0].split('.')[0])

            if class_idx == 1:
                val = np.max(fullmask[:,:,1]) + 1
                
                for link in thing['links']:
                    subobj = gt_dict['detections'][link]

                    for m, uplink in enumerate(subobj['links']):
                        if uplink == key:
                            subclass_idx = int(subobj['class'][m+1].split('.')[1])

                    coords = rp_dict[int(link)].coords
                    point = coords[len(coords)//2]

                    if subclass_idx == 1:
                        idx = mask[int(point[0]), int(point[1])]
                        fullmask[:,:,1][mask == idx] = val
                        fullmask[:,:,5][mask == idx] = val

                    elif subclass_idx == 2:
                        idx = mask[int(point[0]), int(point[1])]
                        fullmask[:,:,2][mask == idx] = val
                        fullmask[:,:,5][mask == idx] = val

            elif class_idx == 2:
                val = np.max(fullmask[:,:,3]) + 1
                
                for link in thing['links']:
                    subobj = gt_dict['detections'][link]

                    for m, uplink in enumerate(subobj['links']):
                        if uplink == key:
                            subclass_idx = int(subobj['class'][m+1].split('.')[1])

                    coords = rp_dict[int(link)].coords
                    point = coords[len(coords)//2]

                    if subclass_idx == 1:
                        idx = mask[int(point[0]), int(point[1])]
                        fullmask[:,:,3][mask == idx] = val
                        fullmask[:,:,6][mask == idx] = val

                    elif subclass_idx == 2:
                        idx = mask[int(point[0]), int(point[1])]
                        fullmask[:,:,4][mask == idx] = val
                        fullmask[:,:,6][mask == idx] = val


        for n in range(fullmask.shape[2]):
            if n == 1 or n == 2 or n == 3 or n == 4:
                continue

            if n == 0:
                category_id = 0
            elif n == 5:
                category_id = 1
            elif n == 6:
                category_id = 2

            boxes = regionprops(fullmask[:,:,n])
            for rp in boxes:
                box = rp.bbox
                box = [box[1], box[0], box[3], box[2]]

                obj = {
                    "bbox": box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [],
                    "category_id": category_id,
                    "iscrowd": 0
                }
                objs.append(obj)
                    
        record["annotations"] = objs
        record["sem_seg"] = fullmask
        dataset_dicts.append(record)

    return dataset_dicts

class MaskDetectionLoader(DatasetMapper):
    def __init__(self, cfg, is_train=True, mask_format="bitmaskmulticlass"):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg
        self.mask_format=mask_format

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Read image and reshape it to always be [h, w, 3].
        image = imread(dataset_dict["file_name"])

        image = np.squeeze(image)

        ### NOT GENERALIZED YET!
        if len(image.shape) > 3:
            image = image[image.shape[0]//2 + np.random.randint(-2,2)]

        if len(image.shape) > 2:
            if image.shape[0] < image.shape[-1]:
                image = np.transpose(image, (1, 2, 0))

            image = image[:,:,0]

        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)

        utils.check_image_size(dataset_dict, image)

        image = image.astype(np.float32)
        if 0 in self.cfg.MODEL.PIXEL_MEAN and 1 in self.cfg.MODEL.PIXEL_STD:
            vallower = np.random.uniform(0.1, 3)
            valupper = np.random.uniform(97,99.9)
            lq, uq = np.percentile(image, [vallower, valupper])
            image = rescale_intensity(image, in_range=(lq,uq), out_range=(0,1))

        mask = dataset_dict['sem_seg'].astype(np.uint16)

        segmap = SegmentationMapsOnImage(mask, shape=image.shape)

        seq = iaa.Sequential([
            iaa.CropToFixedSize(width=self.cfg.INPUT.CROP_SIZE, height=self.cfg.INPUT.CROP_SIZE),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90(k=(0, 3)),
        ])

        image, segmap = seq(image=image, segmentation_maps=segmap)

        image_shape = image.shape[:2]  # h, w
            
        # Convert image to tensor for pytorch model.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        # Convert masks back to detectron2 annotation format.
        segmap = segmap.get_arr()

        # Convert boxes back to detectron2 annotation format.
        annos = []

        for n in range(segmap.shape[2]):
            if n == 1 or n == 2 or n == 3 or n == 4:
                    continue

            boxes = regionprops(segmap[:,:,n])
            for rp in boxes:
                singlemask = np.zeros((segmap.shape[0], segmap.shape[1]), dtype=np.uint8)

                box = rp.bbox

                if n == 0:
                    singlemask[segmap[:,:,n] == rp.label] = 1
                    category_id = 0
                elif n == 5:
                    singlemask[int(box[0]):int(box[2]),int(box[1]):int(box[3])][segmap[:,:,n][int(box[0]):int(box[2]),int(box[1]):int(box[3])] > 0] = 1
                    singlemask[segmap[:,:,1] == rp.label] = 2
                    singlemask[segmap[:,:,2] == rp.label] = 3
                    category_id = 1
                elif n == 6:
                    singlemask[int(box[0]):int(box[2]),int(box[1]):int(box[3])][segmap[:,:,n][int(box[0]):int(box[2]),int(box[1]):int(box[3])] > 0] = 1
                    singlemask[segmap[:,:,3] == rp.label] = 4
                    singlemask[segmap[:,:,4] == rp.label] = 5
                    category_id = 2

                obj = {
                    "bbox": [box[1], box[0], box[3], box[2]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": singlemask,
                    "category_id":  category_id,
                    "iscrowd": 0
                }
                annos.append(obj)

        # Convert bounding box annotations to instances.
        instances = annotations_to_instances(
            annos, image_shape, mask_format=self.mask_format
        )
        
        dataset_dict["instances"] = utils.filter_empty_instances(instances, by_mask=False)

        return dataset_dict

