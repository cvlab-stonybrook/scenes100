#!python3

import os
import sys
import copy
from typing import Dict, List, Optional

import torch

import detectron2
from detectron2.structures import ImageList, Instances

sys.path.append(os.path.join(os.path.dirname(__file__)))


class FPNEarlyFusion(detectron2.modeling.backbone.FPN):
    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.backbone.FPN), 'net is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.bottom_up, detectron2.modeling.backbone.ResNet), 'only support detectron2.modeling.backbone.ResNet backbone'
        input_conv = net.bottom_up.stem.conv1
        # expect: 3 -> 64, 7x7, stride 2x2, padding 3x3, no bias
        assert input_conv.bias is None and input_conv.in_channels == 3 and input_conv.out_channels == 64 and list(input_conv.kernel_size) == [7, 7] and list(input_conv.stride) == [2, 2] and list(input_conv.padding) == [3, 3]
        input_conv.in_channels = 6
        input_conv.weight.data = torch.cat([input_conv.weight.data, copy.deepcopy(input_conv.weight.data)], dim=1) / 2.0 # duplicate conv weights
        net.__class__ = FPNEarlyFusion
        return net


class GeneralizedRCNNEarlyFusion(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        '''
        Normalize, pad and batch the input images.
        '''
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [torch.cat([(x[0:3] - self.pixel_mean) / self.pixel_std, (x[3:6] - self.pixel_mean) / self.pixel_std], dim=0) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.__class__ = GeneralizedRCNNEarlyFusion
        net.backbone = FPNEarlyFusion.create_from_sup(net.backbone)
        return net


class GeneralizedRCNNMidFusion(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training: return self.inference(batched_inputs)
        images_orig, images_diff = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features_orig, features_diff = self.backbone(images_orig.tensor), self.backbone(images_diff.tensor)
        proposals_orig, proposal_losses_orig = self.proposal_generator(images_orig, features_orig, gt_instances)
        _, detector_losses_orig = self.roi_heads(images_orig, features_orig, proposals_orig, gt_instances)
        if self.vis_period > 0:
            raise Exception('visualization of multi-task training not supported')

        features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
        proposals_merge, proposal_losses_merge = self.proposal_generator_merge(images_orig, features_merge, gt_instances)
        _, detector_losses_merge = self.roi_heads_merge(images_orig, features_merge, proposals_merge, gt_instances)

        losses = {}
        losses.update({k: detector_losses_orig[k] * (1 - self.loss_alpha) + detector_losses_merge[k] * self.loss_alpha for k in detector_losses_orig})
        losses.update({k: proposal_losses_orig[k] * (1 - self.loss_alpha) + proposal_losses_merge[k] * self.loss_alpha for k in proposal_losses_orig})
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True, return_both: bool = False, single_image: bool = False):
        # default: only compute & return results from merged features
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        images_orig, images_diff = self.preprocess_image(batched_inputs)
        if single_image:
            assert len(batched_inputs) == 1, 'only supports single image inference'
            stacked = torch.cat([images_orig.tensor, images_diff.tensor], dim=0)
            features_stacked = self.backbone(stacked)
            features_orig = {k: features_stacked[k][0:1] for k in features_stacked}
            features_diff = {k: features_stacked[k][1:2] for k in features_stacked}
        else:
            features_orig, features_diff = self.backbone(images_orig.tensor), self.backbone(images_diff.tensor)
        assert detected_instances is None, 'pre-computed instances not supported'
        if return_both:
            proposals_orig, _ = self.proposal_generator(images_orig, features_orig, None)
            results_orig, _ = self.roi_heads(images_orig, features_orig, proposals_orig, None)

        features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
        proposals_merge, _ = self.proposal_generator_merge(images_orig, features_merge, None)
        results_merge, _ = self.roi_heads_merge(images_orig, features_merge, proposals_merge, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
            if return_both:
                results_orig = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results_orig, batched_inputs, images_orig.image_sizes)
            results_merge = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results_merge, batched_inputs, images_orig.image_sizes)
        if return_both:
            return results_orig, results_merge
        else:
            return results_merge

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [x['image'].to(self.device) for x in batched_inputs]
        images_orig = ImageList.from_tensors([(x[0:3] - self.pixel_mean) / self.pixel_std for x in images], self.backbone.size_divisibility)
        images_diff = ImageList.from_tensors([(x[3:6] - self.pixel_mean) / self.pixel_std for x in images], self.backbone.size_divisibility)
        return images_orig, images_diff

    @staticmethod
    def create_from_sup(net, loss_alpha):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.proposal_generator_merge, net.roi_heads_merge = copy.deepcopy(net.proposal_generator), copy.deepcopy(net.roi_heads)
        net.loss_alpha = loss_alpha
        net.__class__ = GeneralizedRCNNMidFusion
        return net


class GeneralizedRCNNLateFusion(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training: return self.inference(batched_inputs)
        images_orig, images_diff = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features_orig, features_diff = self.backbone(images_orig.tensor), self.backbone(images_diff.tensor)
        proposals_orig, proposal_losses = self.proposal_generator(images_orig, features_orig, gt_instances)
        _, detector_losses_orig = self.roi_heads(images_orig, features_orig, proposals_orig, gt_instances)
        if self.vis_period > 0:
            raise Exception('visualization of multi-task training not supported')

        features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
        _, detector_losses_merge = self.roi_heads_merge(images_orig, features_merge, proposals_orig, gt_instances)

        losses = {}
        losses.update(proposal_losses)
        losses.update({k: detector_losses_orig[k] * (1 - self.loss_alpha) + detector_losses_merge[k] * self.loss_alpha for k in detector_losses_orig})
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True, return_both: bool = False, single_image: bool = False):
        # default: only compute & return results from merged features
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        images_orig, images_diff = self.preprocess_image(batched_inputs)
        if single_image:
            assert len(batched_inputs) == 1, 'only supports single image inference'
            stacked = torch.cat([images_orig.tensor, images_diff.tensor], dim=0)
            features_stacked = self.backbone(stacked)
            features_orig = {k: features_stacked[k][0:1] for k in features_stacked}
            features_diff = {k: features_stacked[k][1:2] for k in features_stacked}
        else:
            features_orig, features_diff = self.backbone(images_orig.tensor), self.backbone(images_diff.tensor)
        assert detected_instances is None, 'pre-computed instances not supported'
        proposals_orig, _ = self.proposal_generator(images_orig, features_orig, None)
        if return_both:
            results_orig, _ = self.roi_heads(images_orig, features_orig, proposals_orig, None)
        features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
        results_merge, _ = self.roi_heads_merge(images_orig, features_merge, proposals_orig, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
            if return_both:
                results_orig = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results_orig, batched_inputs, images_orig.image_sizes)
            results_merge = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results_merge, batched_inputs, images_orig.image_sizes)
        if return_both:
            return results_orig, results_merge
        else:
            return results_merge

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [x['image'].to(self.device) for x in batched_inputs]
        images_orig = ImageList.from_tensors([(x[0:3] - self.pixel_mean) / self.pixel_std for x in images], self.backbone.size_divisibility)
        images_diff = ImageList.from_tensors([(x[3:6] - self.pixel_mean) / self.pixel_std for x in images], self.backbone.size_divisibility)
        return images_orig, images_diff

    @staticmethod
    def create_from_sup(net, loss_alpha):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.roi_heads_merge = copy.deepcopy(net.roi_heads)
        net.loss_alpha = loss_alpha
        net.__class__ = GeneralizedRCNNLateFusion
        return net


if __name__ == '__main__':
    pass
