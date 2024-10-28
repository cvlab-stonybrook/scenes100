import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms

from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from models.dino.dino import DINO, SetCriterion, PostProcess, build_dino
from models.dino.matcher import HungarianMatcher
from models.dino.backbone import Joiner
from models.dino.position_encoding import PositionEmbeddingSineHW
from models.dino.deformable_transformer import build_deformable_transformer 
from models.dino.matcher import build_matcher
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.misc import NestedTensor
from cfg_to_args import cfg_to_args
import copy
__all__ = ["Dino"]


@META_ARCH_REGISTRY.register()
class Dino(nn.Module):
    """
    DINO wrapper for detectron
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        args = cfg_to_args(cfg)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        
        num_classes = args.num_classes
        device = torch.device(args.device)

        model, criterion, postprocessors = build_dino(args)

        model.to(self.device)
        criterion.to(self.device)

        self.model = model
        self.criterion = criterion
        self.to(self.device)
        self.num_select = args.num_select


    def forward(self, batched_inputs, return_features=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
        else:
            targets = None
        
        output = self.model(images, targets, return_features)
        if return_features:
            return output

        if self.training:
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []

            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
                ratio = input_per_image['image'].shape[1] / height
                # breakpoint()
                # self.draw(input_per_image['image']/255, r.pred_boxes.tensor.cpu(), ratio=ratio, file_name=f'test.png')
                # breakpoint()
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)

        num_select = self.num_select
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // box_cls.shape[2]
        labels = topk_indexes % box_cls.shape[2]
        box_pred = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        results = []
        # # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            # threshold = 0.3
            # select_mask = scores_per_image > threshold
            # scores_per_image = scores_per_image[select_mask]
            # labels_per_image = labels_per_image[select_mask]
            # box_pred_per_image = box_pred_per_image[select_mask]

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results   

    def inference_split(self, inputs):
        assert self.training == False, "model not in eval mode"
        # inputs: a single image dict (not batched)
        patch_inputs, offsets = split_into_four(inputs)
        # ratio = inputs['image'].shape[1] / inputs['height']
        # breakpoint()

        with torch.no_grad():
            all_bboxes = []
            all_scores = []
            all_labels = []
            num_instances_from_patches = 0
            for i, patch_input in enumerate(patch_inputs):
                offset = offsets[i]
                instances = self.forward([patch_input])[0]['instances'].to('cpu')
                if i < 4:
                    num_instances_from_patches += instances.pred_boxes.tensor.shape[0]
                    for j in range(instances.pred_boxes.tensor.shape[0]):
                        instances.pred_boxes.tensor[j][0] /= 2
                        instances.pred_boxes.tensor[j][1] /= 2
                        instances.pred_boxes.tensor[j][2] /= 2
                        instances.pred_boxes.tensor[j][3] /= 2
                        instances.pred_boxes.tensor[j][0] += offset[0]
                        instances.pred_boxes.tensor[j][1] += offset[1]
                        instances.pred_boxes.tensor[j][2] += offset[0]
                        instances.pred_boxes.tensor[j][3] += offset[1]
                    
                all_bboxes.append(instances.pred_boxes.tensor)
                all_scores.append(instances.scores)
                all_labels.append(instances.pred_classes)
            # breakpoint()
            all_bboxes = torch.cat(all_bboxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # remove boxes on the border
            keep = remove_splitted_boxes(all_bboxes, inputs['height'], inputs['width'], num_instances_from_patches)
            all_bboxes = all_bboxes[keep]
            all_scores = all_scores[keep]
            all_labels = all_labels[keep]

            # nms
            keep = batched_nms(all_bboxes, all_scores, all_labels, 0.5)
            all_bboxes = all_bboxes[keep]
            all_scores = all_scores[keep]
            all_labels = all_labels[keep]

            # score filtering
            if len(all_scores) > 300:
                _, keep = torch.topk(all_scores, 300)
                all_bboxes = all_bboxes[keep]
                all_scores = all_scores[keep]
                all_labels = all_labels[keep]

            boxes = Boxes(all_bboxes)
            all_instances_dict = {'pred_boxes': boxes, 'scores': all_scores, 'pred_classes': all_labels}
            all_instances = Instances(instances.image_size)
            for (k, v) in all_instances_dict.items():
                all_instances.set(k, v)

            # Dino.draw(inputs['image']/255, all_instances, ratio, 'combine.png')
            # breakpoint()
            return {'instances': all_instances}

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)

        N, _, H, W = images.tensor.shape
        mask = torch.ones((N, H, W), dtype=torch.bool, device=self.device)
        for img_idx, (h, w) in enumerate(images.image_sizes):
            mask[img_idx, : h, : w] = 0
        # images.mask = mask
        nested_tensor = NestedTensor(images.tensor, mask)
        nested_tensor.image_sizes = images.image_sizes
        return nested_tensor
    
    @staticmethod
    def draw(image_tensor, detection_labels, ratio=None, file_name="test_preprocess.png"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        # Convert the PyTorch tensor to a NumPy array for visualization
        image_tensor = image_tensor.cpu()
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image_np)

        # Process the detection labels and draw bounding boxes
        assert ratio is not None, "ratio unavailable"
        detection_bboxes = detection_labels.pred_boxes.tensor.cpu()
        for i, detection in enumerate(detection_bboxes):
            if detection_labels.pred_classes[i] not in [0, 1]:
                continue
            x1, y1, x2, y2 = detection
            x1 *= ratio
            x2 *= ratio
            y1 *= ratio
            y2 *= ratio

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.savefig(file_name, dpi=1200)


def remove_splitted_boxes(all_bboxes, h, w, limit):
    keep = []
    for i in range(all_bboxes.shape[0]):
        if i >= limit:
            keep.append(i)
        elif abs(all_bboxes[i][0] - w/2) > 5 and abs(all_bboxes[i][1] - h/2) > 5 and abs(all_bboxes[i][2] - w/2) > 5 and abs(all_bboxes[i][3] - h/2) > 5:
            keep.append(i)
    return keep


def split_into_four(inputs):
    H_orig, W_orig = inputs['height'], inputs['width']
    img_tensor = inputs['image']
    # img_tensor shape: (3, H, W)
    # Split the image tensor into four parts
    H, W = img_tensor.shape[1:]
    part_height = H // 2
    part_width = W // 2

    part1 = img_tensor[:, :part_height, :part_width]
    part2 = img_tensor[:, :part_height, part_width:]
    part3 = img_tensor[:, part_height:, :part_width]
    part4 = img_tensor[:, part_height:, part_width:]

    # Resize each part to the original size
    part1_resized = F.interpolate(part1.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    part2_resized = F.interpolate(part2.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    part3_resized = F.interpolate(part3.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    part4_resized = F.interpolate(part4.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)

    patches = [part1_resized, part2_resized, part3_resized, part4_resized]
    all_inputs = []
    for patch in patches:
        patch_inputs = {}
        patch_inputs['image'] = patch
        patch_inputs['height'] = inputs['height']
        patch_inputs['width'] = inputs['width']
        if 'video_id' in inputs:
            patch_inputs['video_id'] = inputs['video_id']
        all_inputs.append(patch_inputs)
    all_inputs.append(inputs)
    offsets = [[0, 0], [W_orig//2, 0], [0, H_orig//2], [W_orig//2, H_orig//2], [0, 0]]
    return all_inputs, offsets