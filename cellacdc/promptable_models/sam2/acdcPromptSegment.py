import os
from collections import defaultdict

from cellacdc.promptable_models.utils import build_combined_mask

import numpy as np
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from cellacdc import myutils
from cellacdc.models.sam2 import model_types, sam_models_path


class AvailableModels:
    values = list(model_types.keys())


class NotParam:
    not_a_param = True


class Model:
    def __init__(self, model_type: AvailableModels = "Large", gpu: bool = True):
        """Promptable SAM2 model.

        Parameters
        ----------
        model_type : AvailableModels, optional
            SAM2 model variant to use. Default is "Large".
        gpu : bool, optional
            Whether to run on GPU if available. Default is True.
        """
        if gpu:
            from cellacdc import is_mac_arm64

            if is_mac_arm64:
                device = "cpu"
            else:
                device = "cuda"
        else:
            device = "cpu"

        config_file, sam_checkpoint = model_types[model_type]
        sam_checkpoint = os.path.join(sam_models_path, sam_checkpoint)
        sam = build_sam2(config_file=config_file, ckpt_path=sam_checkpoint, device=device)

        self.model = SAM2ImagePredictor(sam)
        self._embedded_img = None

        self.prompt_ids_image_mapper = {}
        self.prompts = defaultdict(list)
        self.negative_prompts = defaultdict(list)

    def _normalize_prompt(self, prompt):
        prompt = tuple(prompt)
        if len(prompt) != 3:
            raise ValueError(
                "Point prompt must be a sequence of 3 coordinates (z, y, x)."
            )
        z, y, x = prompt
        if z is None or (isinstance(z, float) and np.isnan(z)):
            z = 0
        return int(z), float(y), float(x)

    def _to_rgb(self, image):
        img = myutils.to_uint8(image)
        if img.ndim == 2:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            if img.shape[-1] == 4:
                img = img[..., :3]
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass
        return img

    def _set_image(self, image):
        img_rgb = self._to_rgb(image)
        try:
            init_embeddings = not np.allclose(img_rgb, self._embedded_img)
        except Exception:
            init_embeddings = True

        if init_embeddings:
            self.model.set_image(img_rgb)
            self._embedded_img = img_rgb

    def _collect_prompts(self, prompt_id, treat_other_objects_as_background):
        pos_prompts = self.prompts.get(prompt_id, [])
        neg_prompts = list(self.negative_prompts.get(0, []))
        neg_prompts.extend(self.negative_prompts.get(prompt_id, []))

        if treat_other_objects_as_background:
            for other_id, other_prompts in self.prompts.items():
                if other_id == prompt_id:
                    continue
                neg_prompts.extend(other_prompts)

        return pos_prompts, neg_prompts

    def _points_for_slice(self, prompts, z):
        coords = []
        labels = []
        num_pos = 0
        for prompt, prompt_type, label in prompts:
            if prompt_type != "point":
                raise ValueError(f"Unsupported prompt type: {prompt_type}")

            z_p, y, x = self._normalize_prompt(prompt)
            if z is not None and z_p != z:
                continue

            coords.append([x, y])
            labels.append(label)
            if label == 1:
                num_pos += 1

        if not coords:
            return None, None, 0

        return np.array(coords), np.array(labels), num_pos

    def add_prompt(
        self,
        prompt,
        prompt_id: int,
        *args,
        image=None,
        image_origin=(0, 0, 0),
        parent_obj_id=0,
        prompt_type="point",
        **kwargs,
    ):
        """Add prompt to model."""
        prompt = self._normalize_prompt(prompt)

        if prompt_id not in self.prompt_ids_image_mapper and prompt_id != 0:
            self.prompt_ids_image_mapper[prompt_id] = (image, image_origin)

        if prompt_id != 0:
            self.prompts[prompt_id].append((prompt, prompt_type))
        elif parent_obj_id != 0:
            self.negative_prompts[parent_obj_id].append((prompt, prompt_type))
        else:
            self.negative_prompts[0].append((prompt, prompt_type))

    def segment(
        self,
        image,
        lab: NotParam = None,
        treat_other_objects_as_background: bool = True,
        *args,
        **kwargs,
    ):
        """Run segmentation using the prompts added."""
        is_rgb_image = image.ndim >= 3 and image.shape[-1] in (3, 4)
        is_z_stack = (image.ndim == 3 and not is_rgb_image) or (image.ndim == 4)

        if is_rgb_image:
            lab_out = np.zeros(image.shape[:-1], dtype=np.uint32)
        else:
            lab_out = np.zeros(image.shape, dtype=np.uint32)

        for prompt_id, (prompt_image, image_origin) in self.prompt_ids_image_mapper.items():
            if prompt_id == 0:
                continue

            if prompt_image is None:
                prompt_image = image

            pos_prompts, neg_prompts = self._collect_prompts(
                prompt_id, treat_other_objects_as_background
            )

            is_prompt_rgb = (
                prompt_image.ndim >= 3 and prompt_image.shape[-1] in (3, 4)
            )
            is_prompt_z_stack = (
                (prompt_image.ndim == 3 and not is_prompt_rgb)
                or (prompt_image.ndim == 4)
            )

            if is_prompt_rgb:
                obj_mask = np.zeros(prompt_image.shape[:-1], dtype=bool)
            else:
                obj_mask = np.zeros(prompt_image.shape, dtype=bool)

            prompts = []
            for prompt, prompt_type in neg_prompts:
                prompts.append((prompt, prompt_type, 0))
            for prompt, prompt_type in pos_prompts:
                prompts.append((prompt, prompt_type, 1))

            if not prompts:
                continue

            if is_prompt_z_stack:
                z_dim = obj_mask.shape[0]
                for z in range(z_dim):
                    point_coords, point_labels, num_pos = self._points_for_slice(
                        prompts, z
                    )
                    if num_pos == 0:
                        continue

                    self._set_image(prompt_image[z])
                    multimask_output = len(point_coords) == 1
                    masks, scores, _ = self.model.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=multimask_output,
                    )
                    mask_idx = np.argmax(scores) if multimask_output else 0
                    mask = masks[mask_idx].astype(bool)
                    obj_mask[z][mask] = True
            else:
                point_coords, point_labels, num_pos = self._points_for_slice(
                    prompts, None
                )
                if num_pos == 0:
                    continue

                self._set_image(prompt_image)
                multimask_output = len(point_coords) == 1
                masks, scores, _ = self.model.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=multimask_output,
                )
                mask_idx = np.argmax(scores) if multimask_output else 0
                mask = masks[mask_idx].astype(bool)
                obj_mask[mask] = True

            if not np.any(obj_mask):
                continue

            z0, y0, x0 = map(int, image_origin)
            if obj_mask.ndim == 2:
                obj_slice = (
                    slice(y0, y0 + obj_mask.shape[0]),
                    slice(x0, x0 + obj_mask.shape[1]),
                )
            else:
                obj_slice = (
                    slice(z0, z0 + obj_mask.shape[0]),
                    slice(y0, y0 + obj_mask.shape[1]),
                    slice(x0, x0 + obj_mask.shape[2]),
                )

            lab_out[obj_slice][obj_mask] = prompt_id

        lab_out = build_combined_mask(lab_out)

        self.prompt_ids_image_mapper = {}
        self.prompts = defaultdict(list)
        self.negative_prompts = defaultdict(list)

        return lab_out


def url_help():
    return "https://github.com/facebookresearch/segment-anything-2"
