#!python3

import os
import sys
import copy
import random
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt

import torch
import detectron2
from detectron2.structures import BoxMode

sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import intersect_ratios, bbox_inside


class DatasetMapperMixup(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = np.array(detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format))

        # location-aware mixup
        if 'mixup_src_images' in dataset_dict and random.uniform(0.0, 1.0) < self.mixup_p:
            mixup_src_dict = dataset_dict['mixup_src_images'][random.randrange(0, len(dataset_dict['mixup_src_images']))]
            src_image = detectron2.data.detection_utils.read_image(mixup_src_dict['file_name'], format=self.image_format)
            assert src_image.shape == image.shape

            src_annotations = mixup_src_dict['annotations']
            random.shuffle(src_annotations)
            src_annotations = src_annotations[: max(1, int(self.mixup_r * len(src_annotations)))]
            for ann in src_annotations:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                if not self.mixup_random_position:
                    x1, y1, x2, y2 = map(int, ann['bbox'])
                    x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
                    image[y1 : y2, x1 : x2] = src_image[y1 : y2, x1 : x2]
                else:
                    x1, y1, x2, y2 = map(int, ann['bbox'])
                    x1, y1, x2, y2 = map(lambda x: 1 if x < 1 else x, [x1, y1, x2, y2])
                    x2, y2 = min(image.shape[1], max(x2, x1 + 1)), min(image.shape[0], max(y2, y1 + 1))
                    x_shift, y_shift = np.random.randint(-1 * x1, image.shape[1] - x2), np.random.randint(-1 * y1, image.shape[0] - y2)
                    image[y1 + y_shift : y2 + y_shift, x1 + x_shift : x2 + x_shift] = src_image[y1 : y2, x1 : x2]
                    ann['bbox'] = [x1 + x_shift, y1 + y_shift, x2 + x_shift, y2 + y_shift]
            annotations_trimmed = []
            for ann in dataset_dict['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                _trim = False
                for ann2 in src_annotations:
                    if intersect_ratios(ann['bbox'], ann2['bbox'])[0] >= self.mixup_overlap_thres or bbox_inside(ann['bbox'], ann2['bbox']):
                        _trim = True
                        break
                if not _trim:
                    annotations_trimmed.append(ann)
            for ann in src_annotations:
                annotations_trimmed.append(ann)
            dataset_dict['annotations'] = annotations_trimmed

        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if 'sem_seg_file_name' in dataset_dict:
            sem_seg_gt = detectron2.data.detection_utils.read_image(dataset_dict.pop('sem_seg_file_name'), 'L').squeeze(2)
        else:
            sem_seg_gt = None
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict['sem_seg'] = torch.as_tensor(sem_seg_gt.astype('long'))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            detectron2.data.detection_utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    @staticmethod
    def create_from_sup(mapper, mixup_p, mixup_r, mixup_overlap_thres, mixup_random_position):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperMixup
        mapper.mixup_p, mapper.mixup_r, mapper.mixup_overlap_thres, mapper.mixup_random_position = mixup_p, mixup_r, mixup_overlap_thres, mixup_random_position
        return mapper


class DatasetMapperFusionMixup(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = np.array(detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format))

        # location-aware mixup
        if 'mixup_src_images' in dataset_dict and random.uniform(0.0, 1.0) < self.mixup_p:
            mixup_src_dict = dataset_dict['mixup_src_images'][random.randrange(0, len(dataset_dict['mixup_src_images']))]
            src_image = detectron2.data.detection_utils.read_image(mixup_src_dict['file_name'], format=self.image_format)
            assert src_image.shape == image.shape
            src_annotations = mixup_src_dict['annotations']
            random.shuffle(src_annotations)
            src_annotations = src_annotations[: max(1, int(self.mixup_r * len(src_annotations)))]
            for ann in src_annotations:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                x1, y1, x2, y2 = map(int, ann['bbox'])
                x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
                image[y1 : y2, x1 : x2] = src_image[y1 : y2, x1 : x2]
            annotations_trimmed = []
            for ann in dataset_dict['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                _trim = False
                for ann2 in src_annotations:
                    if intersect_ratios(ann['bbox'], ann2['bbox'])[0] >= self.mixup_overlap_thres or bbox_inside(ann['bbox'], ann2['bbox']):
                        _trim = True
                        break
                if not _trim:
                    annotations_trimmed.append(ann)
            for ann in src_annotations:
                annotations_trimmed.append(ann)
            dataset_dict['annotations'] = annotations_trimmed
        detectron2.data.detection_utils.check_image_size(dataset_dict, image)

        # additional channels using background image
        image_background = detectron2.data.detection_utils.read_image(dataset_dict['file_name_background'], format=self.image_format)
        assert image_background.shape == image.shape
        image, _, image_diff = construct_image_w_background(image, image_background)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        sem_seg_gt = None
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_diff = transforms.apply_image(image_diff)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = np.concatenate([image, image_diff], axis=2)
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    @staticmethod
    def create_from_sup(mapper, mixup_p, mixup_r, mixup_overlap_thres):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperFusionMixup
        mapper.mixup_p, mapper.mixup_r, mapper.mixup_overlap_thres = mixup_p, mixup_r, mixup_overlap_thres
        return mapper


def construct_image_w_background(image, image_background):
    image_diff = (image.astype(np.float16) - image_background) # float16, [-255, 255]
    image_diff = ((image_diff + 255) * 0.5).astype(np.uint8)
    return image, image_background, image_diff


if __name__ == '__main__':
    pass
