#!python3

import os
import sys
import types
import time
import gc
import json
import copy
import random
import tqdm
import glob
import hashlib
import contextlib
import argparse

from PIL import Image, ImageDraw, ImageFont
import multiprocessing
from multiprocessing import Pool as ProcessPool
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata
import torchvision

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances, Boxes

import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from adaptation.mscoco_remap_dataset import get_coco_dicts
from adaptation.evaluator import evaluate_masked


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_inference_server')


def get_cfg_base_model(m, ckpt=None):
    models = {
        'r18-fpn-3x':  os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'faster_rcnn_R_18_FPN_3x.yaml'),
        'r101-fpn-3x': model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'),
    }
    assert m in models, 'model %s not recognized' % m

    cfg = get_cfg()
    cfg.merge_from_file(models[m])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    if not ckpt is None:
        assert os.access(ckpt, os.R_OK), '%s not readable' % ckpt
        cfg.MODEL.WEIGHTS = ckpt
        cfg.MODEL.WEIGHTS = os.path.normpath(cfg.MODEL.WEIGHTS)

    print('detectron2 model:', m)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    return cfg


class GeneralizedRCNNServer(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def get_training_assignment(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        for im in batched_inputs:
            if im['video_id'] != 'coco' and im['video_id'] not in self.video_id_to_index:
                if len(self.un_used_indices) > 0:
                    i = sorted(list(self.un_used_indices.keys()))[0]
                    self.video_id_to_index[im['video_id']] = i
                    del self.un_used_indices[i]
                    self.used_indices[i] = True
                else:
                    self.video_id_to_index[im['video_id']] = np.random.choice(list(self.used_indices.keys()))
                print('assign new video %s -> %s' % (im['video_id'], self.video_id_to_index[im['video_id']]))
        module_indices = []
        for im in batched_inputs:
            # randomly train 1 path for COCO images
            if im['video_id'] == 'coco':
                if len(self.used_indices) > 0:
                    module_indices.append(np.random.choice(list(self.used_indices.keys())))
                else:
                    module_indices.append(np.random.choice(list(self.un_used_indices.keys())))
            else:
                module_indices.append(self.video_id_to_index[im['video_id']])
        return batched_inputs, module_indices

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        # assign video ID to module
        batched_inputs, module_indices = self.get_training_assignment(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert 'instances' in batched_inputs[0], 'ground truth missing'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]

        # FPN
        feature_res2 = [self.stem_res2_list[m](images.tensor[i : i + 1]) for i, m in enumerate(module_indices)]
        feature_res2 = torch.cat(feature_res2, dim=0)
        features = self.backbone(feature_res2)

        # RPN
        proposals, proposal_losses = [], {}
        for i, m in enumerate(module_indices):
            p, pl = self.proposal_generator_list[m](
                ImageList(images.tensor[i : i + 1], images.image_sizes[i : i + 1]),
                {f: features[f][i : i + 1] for f in features},
                gt_instances[i : i + 1]
            )
            proposals.append(p[0])
            for k in pl:
                if k in proposal_losses:
                    proposal_losses[k] += pl[k]
                else:
                    proposal_losses[k] = pl[k]
        for k in proposal_losses:
            proposal_losses[k] /= len(batched_inputs)

        # ROI
        proposals_roi_sampled = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
        box_features = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.box_in_features], [x.proposal_boxes for x in proposals_roi_sampled])
        box_features = self.roi_heads.box_head(box_features)

        detector_losses = {}
        for p, m in zip(proposals_roi_sampled, module_indices):
            predictions_m = self.box_predictor_list[m](box_features[: len(p)])
            dl = self.box_predictor_list[m].losses(predictions_m, [p])
            assert not self.roi_heads.train_on_pred_boxes
            for k in dl:
                if k in detector_losses:
                    detector_losses[k] += dl[k]
                else:
                    detector_losses[k] = dl[k]
            box_features = box_features[len(p) :]
        del box_features
        for k in detector_losses:
            detector_losses[k] /= len(batched_inputs)

        if self.vis_period > 0:
            raise Exception('visualization of multi-task training not supported')
            storage = detectron2.utils.events.get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        assert detected_instances is None

        # pre-process
        module_indices = []
        for im in batched_inputs:
            # at inference time, assign first module for all unseen video IDs
            if im['video_id'] == 'coco' or im['video_id'] not in self.video_id_to_index:
                module_indices.append(0)
            else:
                module_indices.append(self.video_id_to_index[im['video_id']])
        images = self.preprocess_image(batched_inputs)

        # FPN
        feature_res2 = [self.stem_res2_list[m](images.tensor[i : i + 1]) for i, m in enumerate(module_indices)]
        feature_res2 = torch.cat(feature_res2, dim=0)
        features = self.backbone(feature_res2)

        # RPN
        proposals = [
            self.proposal_generator_list[m](
                ImageList(images.tensor[i : i + 1], images.image_sizes[i : i + 1]),
                {f: features[f][i : i + 1] for f in features},
                None
            )[0][0]
            for i, m in enumerate(module_indices)
        ]

        # ROI
        box_features = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.box_in_features], [x.proposal_boxes for x in proposals])
        box_features = self.roi_heads.box_head(box_features)

        results = []
        for p, m in zip(proposals, module_indices):
            predictions_m = self.box_predictor_list[m](box_features[: len(p)])
            results.extend(self.box_predictor_list[m].inference(predictions_m, [p])[0])
            box_features = box_features[len(p) :]
        del box_features

        if do_postprocess:
            assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
            return detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    @classmethod
    def create_from_sup(cls, net, budget):
        # keep budget # of paths, using a dict to map video_id to list
        net.budget = budget
        print('module budget:', budget)
        net.video_id_to_index = {}
        net.used_indices = {}
        net.un_used_indices = {i: True for i in range(0, net.budget)}
        net.stem_res2_list = torch.nn.ModuleList([
            torch.nn.Sequential(
                copy.deepcopy(net.backbone.bottom_up.stem),
                copy.deepcopy(net.backbone.bottom_up.res2)
            ) for _ in range(0, net.budget)
        ])
        del net.backbone.bottom_up.stem
        net.backbone.bottom_up.res2 = torch.nn.Identity()
        net.backbone.bottom_up.stages[net.backbone.bottom_up.stage_names.index('res2')] = torch.nn.Identity()
        assert isinstance(net.backbone.bottom_up, detectron2.modeling.backbone.resnet.ResNet)
        net.backbone.bottom_up.__class__ = ResNetNoStemRes2
        net.proposal_generator_list = torch.nn.ModuleList([copy.deepcopy(net.proposal_generator) for _ in range(0, net.budget)])
        del net.proposal_generator
        net.box_predictor_list = torch.nn.ModuleList([copy.deepcopy(net.roi_heads.box_predictor) for _ in range(0, net.budget)])
        del net.roi_heads.box_predictor
        net.__class__ = cls
        return net


class ResNetNoStemRes2(detectron2.modeling.backbone.resnet.ResNet):
    def forward(self, x):
        assert x.dim() == 4
        outputs = {}
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs


def faster_rcnn_inference_return_feature(frcnn: detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN, batched_inputs: List[Dict[str, torch.Tensor]]):
    assert not frcnn.training
    images = frcnn.preprocess_image(batched_inputs)
    features = frcnn.backbone(images.tensor)
    # apply adaptive avg pooling on p3/p4/p5 features, used for buffering later
    f_fpn = torch.cat([
        # torch.nn.functional.adaptive_avg_pool2d(features['p3'], (3, 5)),
        torch.nn.functional.adaptive_max_pool2d(features['p3'], (3, 5)),
        torch.nn.functional.adaptive_max_pool2d(features['p4'], (3, 5)),
        torch.nn.functional.adaptive_max_pool2d(features['p5'], (3, 5))
        ], dim=1
    ).view(len(batched_inputs), -1).detach()
    proposals, _ = frcnn.proposal_generator(images, features, None)
    results, _ = frcnn.roi_heads(images, features, proposals, None)
    assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
    return detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes), f_fpn


class AdaptativePartialTrainer(DefaultTrainer):
    def __init__(self, cfg, train_whole):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        self.model_teacher = copy.deepcopy(model)
        self.model_teacher.eval()
        model = GeneralizedRCNNServer.create_from_sup(model, cfg.MODEL.ADAPTIVE_BUDGET)

        if train_whole:
            trainable_modules = [model]
        else:
            trainable_modules = [
                model.stem_res2_list,
                model.proposal_generator_list,
                model.box_predictor_list,
            ]
        _count_all, _count_train = 0, 0
        for p in model.parameters():
            _count_all += p.numel()
            p.requires_grad = False
        for m in trainable_modules:
            for p in m.parameters():
                _count_train += p.numel()
                p.requires_grad = True
        print('only train subset of model parameters: %d/%d %.4f%%' % (_count_train, _count_all, _count_train / _count_all * 100))
        optimizer = self.build_optimizer(cfg, torch.nn.ModuleList(trainable_modules))
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        self._trainer.model_teacher = self.model_teacher
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []
        self._trainer.pseudo_det_min_score = cfg.SOLVER.PSEUDO_DET_MIN_SCORE

    def build_hooks(self):
        ret = super().build_hooks()
        self.eval_results_all = {}
        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = copy.deepcopy(self._last_eval_results)
            return self._last_eval_results
        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)

        def save_model_state():
            model = self.model
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                print('unwrap data parallel')
                model = model.module
            prefix = '%s.iter.%d' % (self.cfg.SOLVER.SAVE_PREFIX, self.iter)
            torch.save(model.state_dict(), prefix + '.pth')
            mapper = {
                'budget': model.budget,
                'video_id_to_index': model.video_id_to_index,
                'used_indices': model.used_indices,
                'un_used_indices': model.un_used_indices
            }
            print(mapper)
            torch.save(mapper, prefix + '.mapper.pth')
            print('saved model state to:', prefix)

        ret.append(detectron2.engine.hooks.EvalHook(self.cfg.SOLVER.SAVE_INTERVAL, save_model_state))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=finetune_output)


def finetune_ema_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start

    pseudo_idx, pseudo_inputs = [], []
    for _i, _d in enumerate(data):
        if 'image_test' in _d:
            pseudo_idx.append(_i)
            _h, _w = _d['instances'].image_size
            pseudo_inputs.append({'image': _d['image_test'], 'height': _h, 'width': _w})
    if len(pseudo_idx) > 0:
        with torch.no_grad():
            pseudo_labels, pseudo_feature = faster_rcnn_inference_return_feature(self.model_teacher, pseudo_inputs)
            for _i, _pred, _f in zip(pseudo_idx, pseudo_labels, pseudo_feature):
                _mask = _pred['instances'].scores >= self.pseudo_det_min_score
                _filtered = Instances(_pred['instances']._image_size)
                _filtered.set('gt_boxes', _pred['instances'].pred_boxes[_mask])
                _filtered.set('gt_classes', _pred['instances'].pred_classes[_mask])
                data[_i]['instances'] = _filtered
                data[_i]['feature_fpn'] = _f.cpu() # reduce VRAM usage
                del data[_i]['image_test']

    loss_dict = self.model(data)
    loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
    if isinstance(loss_dict, torch.Tensor):
        losses = loss_dict
        loss_dict = {'total_loss': loss_dict}
    else:
        losses = sum(loss_dict.values())

    self.optimizer.zero_grad()
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    # If you need gradient clipping/scaling or other processing, you can wrap the optimizer with your custom `step()` method. But it is suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    self.optimizer.step()

    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


class DatasetMapperPseudo(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
        # for generating pseudo labels on the fly
        if 'source' in dataset_dict and dataset_dict['source'] == 'unlabeled':
            image_test = self.apply_test_transform(image)
            if image_test is not None:
                dataset_dict['image_test'] = torch.as_tensor(np.ascontiguousarray(image_test.transpose(2, 0, 1)))
        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        assert self.proposal_topk is None
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    def apply_test_transform(self, image):
        if not (image.dtype == np.uint8 and len(image.shape) == 3 and image.shape[2] == 3):
            return None
        h, w = image.shape[:2]
        newh, neww = self.get_output_shape(h, w, self.min_size_teacher, self.max_size_teacher)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
        return np.asarray(pil_image)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    @staticmethod
    def create_from_sup(mapper, cfg):
        assert not cfg.INPUT.CROP.ENABLED
        assert cfg.INPUT.RANDOM_FLIP == 'none'
        mapper.min_size_teacher, mapper.max_size_teacher = cfg.INPUT.MIN_SIZE_TEACHER, cfg.INPUT.MAX_SIZE_TEACHER
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperPseudo
        return mapper


def adapt(args):
    random.seed(42)

    # validation images
    desc_manual_valid, dst_manual_valid = 'scenes100_manual', []
    for v in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'annotation', v)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        for i in range(0, len(annotations)):
            annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
            annotations[i]['video_id'] = v
        print('manual annotation for %s: %d images, %d bboxes' % (v, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
        dst_manual_valid.extend(annotations)
    for i in range(0, len(dst_manual_valid)):
        assert 'video_id' in dst_manual_valid[i]
        dst_manual_valid[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(dst_manual_valid), sum(list(map(lambda x: len(x['annotations']), dst_manual_valid)))))

    # training images
    dst_cocotrain = get_coco_dicts('train')
    for im in dst_cocotrain:
        im['source'] = 'coco'
        im['video_id'] = 'coco'
    random.shuffle(dst_cocotrain)

    dst_pseudo_anno, desc_pseudo_anno = [], 'scenes100_pseudo_cocotrain'
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    for v in video_id_list:
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'images', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = meta['ifilelist']
        dict_json = []
        for i in range(0, len(ifilelist)):
            dict_json.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}], 'source': 'unlabeled', 'video_id': v})
        print('unlabeled frames of video %s at %s: %d images' % (v, lmdb_path, len(dict_json)))
        if len(dict_json) > images_per_video_cap:
            random.shuffle(dict_json)
            dict_json = dict_json[:images_per_video_cap]
            print('randomly downsampled to: %d images' % len(dict_json))
        dst_pseudo_anno.extend(dict_json)
    print('total unlabeled: %d images' % len(dst_pseudo_anno))

    # 1/6 of images are from MSCOCO
    dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[: len(dst_pseudo_anno) // 5]
    for i in range(0, len(dst_pseudo_anno)):
        assert 'video_id' in dst_pseudo_anno[i]
        dst_pseudo_anno[i]['image_id'] = i + 1
    print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
    prefix = 'adapt_server_%s_%s.%s' % (args.model, desc_pseudo_anno, args.tag)

    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    cfg = get_cfg_base_model(args.model)
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.INPUT.MIN_SIZE_TEACHER = int(args.teacher_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEACHER = int(args.teacher_scale * cfg.INPUT.MAX_SIZE_TEST)
    cfg.INPUT.MIN_SIZE_TEST = int(args.student_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.student_scale * cfg.INPUT.MAX_SIZE_TEST)
    cfg.INPUT.MIN_SIZE_TRAIN = tuple([int(args.student_scale * s) for s in cfg.INPUT.MIN_SIZE_TRAIN])
    cfg.INPUT.MAX_SIZE_TRAIN = int(args.student_scale * cfg.INPUT.MAX_SIZE_TRAIN)

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 200
    cfg.SOLVER.GAMMA = 1
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid,)

    cfg.SOLVER.PSEUDO_DET_MIN_SCORE = args.refine_det_score_thres
    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    cfg.SOLVER.SAVE_INTERVAL = args.save_interval
    cfg.SOLVER.SAVE_PREFIX = os.path.join(args.outputdir, prefix)
    cfg.DATASETS.TRAIN_NUM_IMAGES = len(dst_pseudo_anno)
    cfg.DATASETS.TEST_NUM_IMAGES = len(dst_manual_valid)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = AdaptativePartialTrainer(cfg, args.train_whole)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_ema_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperPseudo.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, cfg)

    # manually load the parameters from ckpt
    print('loading weights from:', args.ckpt)
    # trainer.resume_or_load(resume=False)
    assert trainer.model is trainer._trainer.model
    assert trainer.model_teacher is trainer._trainer.model_teacher
    state_dict = torch.load(args.ckpt)
    if args.ckpt_teacher is None:
        trainer.model_teacher.load_state_dict(state_dict)
    else:
        state_dict_teacher = torch.load(args.ckpt_teacher)
        trainer.model_teacher.load_state_dict(state_dict_teacher)
        del state_dict_teacher
        print('loaded teacher weights from:', args.ckpt_teacher)
    keys_del = []
    for k in tqdm.tqdm(list(state_dict.keys()), ascii=True, desc='populating weights for sub-modules'):
        if k.startswith('backbone.bottom_up.stem.'):
            for i in range(0, args.budget):
                state_dict['stem_res2_list.%d.0.%s' % (i, k[24:])] = copy.deepcopy(state_dict[k])
        if k.startswith('backbone.bottom_up.res2.'):
            for i in range(0, args.budget):
                state_dict['stem_res2_list.%d.1.%s' % (i, k[24:])] = copy.deepcopy(state_dict[k])
        if k.startswith('proposal_generator.'):
            for i in range(0, args.budget):
                state_dict['proposal_generator_list.%d.%s' % (i, k[19:])] = copy.deepcopy(state_dict[k])
        if k.startswith('roi_heads.box_predictor.'):
            for i in range(0, args.budget):
                state_dict['box_predictor_list.%d.%s' % (i, k[24:])] = copy.deepcopy(state_dict[k])
    for k in state_dict.keys():
        if k.startswith('backbone.bottom_up.stem.') or k.startswith('backbone.bottom_up.res2.') or k.startswith('proposal_generator.') or k.startswith('roi_heads.box_predictor.'):
            keys_del.append(k)
    for k in keys_del:
        del state_dict[k]
    trainer.model.load_state_dict(state_dict)
    del state_dict
    if args.resume_prefix is not None:
        print('resume from:', args.resume_prefix)
        mapper = torch.load(args.resume_prefix + '.mapper.pth')
        print(mapper)
        assert mapper['budget'] == trainer.model.budget, 'budget mismatch'
        trainer.model.video_id_to_index = mapper['video_id_to_index']
        trainer.model.used_indices = mapper['used_indices']
        trainer.model.un_used_indices = mapper['un_used_indices']
        state_dict = torch.load(args.resume_prefix + '.pth')
        trainer.model.load_state_dict(state_dict)
        del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    print(prefix)
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(list(aps.keys()))
    dst_list = {'mAP': [], 'AP50': []}
    for i in iter_list:
        dst_list['mAP'].append(aps[i]['bbox']['AP'])
        dst_list['AP50'].append(aps[i]['bbox']['AP50'])

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for x in loss_history:
        for loss_key in x['loss']:
            if not loss_key in loss_history_dict:
                loss_history_dict[loss_key] = []
            loss_history_dict[loss_key].append([x['iter'], x['loss'][loss_key]])
    loss_history_dict = {loss_key: np.array(loss_history_dict[loss_key]) for loss_key in loss_history_dict}
    for loss_key in loss_history_dict:
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Valid AP50', 'Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')

    plt.subplot(1, 2, 2)
    colors, color_i = ['#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#000000'], 0
    legends = []
    for loss_key in loss_history_dict:
        plt.plot(loss_history_dict[loss_key][:, 0], loss_history_dict[loss_key][:, 1], linestyle='-', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, prefix + '.pdf'))


def convert_budget1_to_single(args):
    mapper = torch.load(os.path.join(args.ckpts_dir, '%s.mapper.pth' % args.ckpts_tag))
    assert mapper['budget'] == 1, 'can only convert budget=1 model'
    state_dict = torch.load(os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    keys_del = []
    for k in tqdm.tqdm(list(state_dict.keys()), ascii=True):
        if k.startswith('stem_res2_list.0.0.'):
            state_dict['backbone.bottom_up.stem.' + k[19:]] = state_dict[k]
            keys_del.append(k)
        if k.startswith('stem_res2_list.0.1.'):
            state_dict['backbone.bottom_up.res2.' + k[19:]] = state_dict[k]
            keys_del.append(k)
        if k.startswith('proposal_generator_list.0.'):
            state_dict['proposal_generator.' + k[26:]] = state_dict[k]
            keys_del.append(k)
        if k.startswith('box_predictor_list.0.'):
            state_dict['roi_heads.box_predictor.' + k[21:]] = state_dict[k]
            keys_del.append(k)
    for k in keys_del:
        del state_dict[k]
    torch.save(state_dict, os.path.join(args.ckpts_dir, '%s.single.pth' % args.ckpts_tag))


class _MapperDataset(torchdata.Dataset):
    def __init__(self, cfg, images):
        super(_MapperDataset, self).__init__()
        self.image_mapper = detectron2.data.DatasetMapper(**detectron2.data.DatasetMapper.from_config(cfg, is_train=False))
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        return self.image_mapper(self.images[i])
    @staticmethod
    def collate(batch):
        return batch

def offline_clustering(args):
    from sklearn.cluster import KMeans

    mapper = torch.load(os.path.join(args.ckpts_dir, '%s.mapper.pth' % args.ckpts_tag))
    assert mapper['budget'] == 1, 'can only run offline clustering from budget=1 model'
    cfg = get_cfg_base_model(args.model)
    model = GeneralizedRCNNServer.create_from_sup(DefaultPredictor(cfg).model, mapper['budget'])
    state_dict = torch.load(os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    model.load_state_dict(state_dict)
    print('loaded weights from:', os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    model.budget = mapper['budget']
    model.video_id_to_index = mapper['video_id_to_index']
    model.used_indices = mapper['used_indices']
    model.un_used_indices = mapper['un_used_indices']
    model.eval()

    video_images, num_images_per_video = [], 400
    for v in tqdm.tqdm(video_id_list, ascii=True, desc='loading images'):
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'images', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = sorted(meta['ifilelist'])
        assert len(ifilelist) / num_images_per_video > 2
        ifilelist = np.array(ifilelist[: (len(ifilelist) // num_images_per_video) * num_images_per_video])
        ifilelist = ifilelist.reshape(-1, num_images_per_video)[0]
        for i in range(0, len(ifilelist)):
            video_images.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': [], 'video_id': v})
    print('total images:', len(video_images))

    dataset = _MapperDataset(cfg, video_images)
    loader = torchdata.DataLoader(dataset, batch_size=6, collate_fn=_MapperDataset.collate, shuffle=False, num_workers=6)

    video_id_features = []
    video_features_p3, video_features_p4, video_features_p5 = [], [], []
    video_features_classes = {i: [] for i in range(0, len(thing_classes))}
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, ascii=True, desc='extracting features'):
            video_id_features.extend([im['video_id'] for im in batch])
            images = model.preprocess_image(batch)
            feature_res2 = model.stem_res2_list[0](images.tensor)
            features = model.backbone(feature_res2)

            # image level features
            features_p3 = torch.nn.functional.adaptive_avg_pool2d(features['p3'], (12, 21)).view(len(batch), -1)
            video_features_p3.append(features_p3.detach().cpu())
            features_p4 = torch.nn.functional.adaptive_avg_pool2d(features['p4'], (9, 16)).view(len(batch), -1)
            video_features_p4.append(features_p4.detach().cpu())
            features_p5 = torch.nn.functional.adaptive_avg_pool2d(features['p5'], (5, 9)).view(len(batch), -1)
            video_features_p5.append(features_p5.detach().cpu())

            proposals, _ = model.proposal_generator_list[0](images, features, None)
            box_features = model.roi_heads.box_pooler([features[f] for f in model.roi_heads.box_in_features], [x.proposal_boxes for x in proposals])
            box_features = model.roi_heads.box_head(box_features)
            predictions  = model.box_predictor_list[0](box_features)

            boxes  = model.box_predictor_list[0].predict_boxes(predictions, proposals)
            scores = model.box_predictor_list[0].predict_probs(predictions, proposals)

            # based on https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/roi_heads/fast_rcnn.py#L126
            for scores_per_image, boxes_per_image, p in zip(scores, boxes, proposals):
                valid_mask = torch.isfinite(boxes_per_image).all(dim=1) & torch.isfinite(scores_per_image).all(dim=1)
                if not valid_mask.all():
                    boxes_per_image = boxes_per_image[valid_mask]
                    scores_per_image = scores_per_image[valid_mask]
                scores_per_image = scores_per_image[:, :-1] # R x C

                boxes_per_image = Boxes(boxes_per_image.reshape(-1, 4))
                boxes_per_image.clip(p.image_size)
                boxes_per_image = boxes_per_image.tensor.view(-1, len(thing_classes), 4) # R x C x 4

                box_features_per_image = box_features[: p.objectness_logits.size(0)] # R x 1024
                box_features = box_features[p.objectness_logits.size(0) :]

                # for each image and each class, mean box feature for top instances, after per-class NMS
                for i in range(0, len(thing_classes)):
                    topk_idx = torchvision.ops.batched_nms(
                        boxes_per_image[:, i, :].float(),
                        scores_per_image[:, i],
                        torch.zeros((scores_per_image.size(0),), dtype=torch.int32),
                        0.5
                    ) # already sorted in decreasing order by scores
                    box_features_per_image_per_class_mean = box_features_per_image[topk_idx].mean(dim=0)
                    video_features_classes[i].append(box_features_per_image_per_class_mean.detach().cpu())

    for k in list(state_dict.keys()):
        if k.startswith('stem_res2_list.0.'):
            for b in range(0, args.budget):
                state_dict['stem_res2_list.' + str(b) + k[16:]] = copy.deepcopy(state_dict[k])
        elif k.startswith('proposal_generator_list.0.'):
            for b in range(0, args.budget):
                state_dict['proposal_generator_list.' + str(b) + k[25:]] = copy.deepcopy(state_dict[k])
        elif k.startswith('box_predictor_list.0.'):
            for b in range(0, args.budget):
                state_dict['box_predictor_list.' + str(b) + k[20:]] = copy.deepcopy(state_dict[k])

    video_id_features = np.array(video_id_features)
    video_features_p3 = torch.cat(video_features_p3, dim=0).detach().numpy()
    video_features_p4 = torch.cat(video_features_p4, dim=0).detach().numpy()
    video_features_p5 = torch.cat(video_features_p5, dim=0).detach().numpy()
    video_features_classes = {i: torch.stack(video_features_classes[i], dim=0).detach().numpy() for i in video_features_classes}
    torch.cuda.empty_cache()
    for features, desc in [
            (video_features_p3, 'fpn.p3'),
            (video_features_p4, 'fpn.p4'),
            (np.concatenate([video_features_p3, video_features_p4], axis=1), 'fpn.p3.p4'),
            (video_features_p5, 'fpn.p5'),
            (np.concatenate([video_features_p4, video_features_p5], axis=1), 'fpn.p4.p5'),
            (video_features_classes[0], 'roi.person'),
            (video_features_classes[1], 'roi.vehicle'),
            (np.concatenate([video_features_classes[0], video_features_classes[1]], axis=1), 'roi.person.vehicle')]:
        print('running %s-Means for %s: %s %s' % (args.budget, desc, features.shape, features.dtype))
        kmeans = KMeans(n_clusters=args.budget, random_state=0).fit(features)

        mapper = {'budget': args.budget, 'video_id_to_index': {}, 'used_indices': {}, 'un_used_indices': {b: True for b in range(0, args.budget)}}
        for v in video_id_list:
            cluster_ids = kmeans.labels_[video_id_features == v]
            i = np.argmax(np.bincount(cluster_ids))
            mapper['video_id_to_index'][v] = i
            mapper['used_indices'][i] = True
            if i in mapper['un_used_indices']:
                del mapper['un_used_indices'][i]

        print(mapper)
        prefix = os.path.join(args.ckpts_dir, '%s.%smeans.%s' % (args.ckpts_tag, args.budget, desc))
        torch.save(state_dict, prefix + '.pth')
        torch.save(mapper, prefix + '.mapper.pth')


class SemiRandomClient(torchdata.Dataset):
    def __init__(self, cfg):
        super(SemiRandomClient, self).__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == 'BGR'

        self.images = []
        for video_id in video_id_list:
            inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'annotation', video_id))
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                _dicts = json.load(fp)
            for im in _dicts:
                im['md5'] = '%s_%s' % (video_id, im['file_name']) # for pseudo-random shuffling
                im['md5'] = hashlib.md5(im['md5'].encode('utf-8')).hexdigest()
                im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))
                im['video_id'] = video_id
            self.images.extend(_dicts)
        self.images.sort(key=lambda x: x['md5'])
        self.preloaded_images = None

    def preload(self):
        if self.preloaded_images is not None:
            return
        self.preloaded_images = []
        for i in tqdm.tqdm(range(0, len(self.images)), ascii=True, desc='preloading images'):
            self.preloaded_images.append(self.read(i))

    def __len__(self):
        return len(self.images)

    def read(self, i):
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        return {'image': image, 'height': height, 'width': width, 'video_id': self.images[i]['video_id']}

    def __getitem__(self, i):
        if self.preloaded_images is None:
            return self.read(i), self.images[i]
        else:
            return self.preloaded_images[i], self.images[i]

    @staticmethod
    def collate(batch):
        return batch

def _run_AP_eval(detections):
    results = {}
    for video_id in detections:
        for det in detections[video_id]:
            # bbox has format [x1, y1, x2, y2]
            bbox, score, label = det['instances']['bbox'].numpy().tolist(), det['instances']['score'].numpy().tolist(), det['instances']['label'].numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
            del det['instances']
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id])
        print(video_id, end=' ', flush=True)
    return results

def simulate(args):
    mapper = torch.load(os.path.join(args.ckpts_dir, '%s.mapper.pth' % args.ckpts_tag))
    index_to_video_id = {}
    for v, i in mapper['video_id_to_index'].items():
        if not i in index_to_video_id:
            index_to_video_id[i] = [v]
        else:
            index_to_video_id[i].append(v)
    for i in sorted(list(index_to_video_id.keys())):
        print(i, sorted(index_to_video_id[i]))
    print('unused:', mapper['un_used_indices'])
    mapped_videos_all = []
    for v_list in index_to_video_id.values():
        mapped_videos_all.extend(v_list)
    mapped_videos_all = set(mapped_videos_all)
    print('all mapped videos:', sorted(list(mapped_videos_all)))
    print('unmapped videos:', sorted(list(set(video_id_list) - mapped_videos_all)))

    cfg = get_cfg_base_model(args.model)
    cfg.INPUT.MIN_SIZE_TEST = int(args.student_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.student_scale * cfg.INPUT.MAX_SIZE_TEST)
    model = GeneralizedRCNNServer.create_from_sup(DefaultPredictor(cfg).model, mapper['budget'])
    model.video_id_to_index = mapper['video_id_to_index']
    model.used_indices = mapper['used_indices']
    model.un_used_indices = mapper['un_used_indices']

    state_dict = torch.load(os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    model.load_state_dict(state_dict)
    print('loaded weights from:', os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    del state_dict

    dataset = SemiRandomClient(cfg)
    if args.preload:
        dataset.preload()
    loader = torchdata.DataLoader(dataset, batch_size=None, collate_fn=SemiRandomClient.collate, shuffle=False, num_workers=8)
    gc.collect()
    torch.cuda.empty_cache()

    detections = {v: [] for v in video_id_list}
    t_total = time.time()
    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='detecting'):
        det = copy.deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            instances = model.inference([inputs])[0]['instances'].to('cpu')
            det['instances'] = {
                'bbox': instances.pred_boxes.tensor,
                'score': instances.scores,
                'label': instances.pred_classes
            }
        detections[im['video_id']].append(det)
    t_total = time.time() - t_total
    print('%d finished in %.1f seconds, throughput %.3f images/sec' % (len(dataset), t_total, len(dataset) / t_total))

    for video_id in detections:
        detections[video_id] = sorted(detections[video_id], key=lambda x: x['file_name'])

    pool = ProcessPool(processes=16)
    args_list = [{v: detections[v]} for v in detections]
    rets = pool.map_async(_run_AP_eval, args_list).get()
    pool.close()
    pool.join()
    print('')
    results = {}
    for r in rets:
        results.update(r)
    assert len(results) == 100

    with open(os.path.join(args.ckpts_dir, '%s.AP.json' % args.ckpts_tag), 'w') as fp:
        json.dump(results, fp)
    categories = ['person', 'vehicle', 'overall', 'weighted']
    aps = {c: [] for c in categories}
    for video_id in results:
        AP = results[video_id]['results']
        for cat in categories:
            aps[cat].append([AP[cat][0], AP[cat][1]])
    print(aps)
    for cat in categories:
        aps[cat] = np.array(aps[cat]) * 100.0
        print('%s: mAP %.4f, AP50 %.4f' % (cat, aps[cat][:, 0].mean(), aps[cat][:, 1].mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, default='server', help='option')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--budget', type=int, help='number of model paths')
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--ckpt_teacher', type=str, default=None, help='weights checkpoint of model used as teacher, use same as ckpt if not set')
    parser.add_argument('--resume_prefix', type=str, default=None, help='resume training from weights and mapper')

    # training setting
    parser.add_argument('--train_whole', type=bool, default=False, help='train the whole network')
    parser.add_argument('--teacher_scale', type=float, default=2, help='scale for teacher input, on top of default image sizes')
    parser.add_argument('--student_scale', type=float, default=1, help='scale for student input, on top of default image sizes')

    parser.add_argument('--ckpts_dir', type=str, default=None, help='weights checkpoints of individual models')
    parser.add_argument('--ckpts_tag', type=str, default='')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)
    parser.add_argument('--preload', type=bool, default=False)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--save_interval', type=int, help='interval for saving model')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    args = parser.parse_args()
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'adapt':
        adapt(args)
    if args.opt == 'convert':
        convert_budget1_to_single(args)
    if args.opt == 'cluster':
        offline_clustering(args)
    if args.opt == 'server':
        simulate(args)


'''
python inference_server_simulate.py --model r101-fpn-3x --ckpts_dir --ckpts_tag
python inference_server_simulate.py --model r101-fpn-3x --instances 2 --ckpt
python inference_server_simulate.py --opt tp --id 001 --model r101-fpn-3x --ckpts_dir --ckpts_tag

python inference_server_simulate.py --opt convert --ckpts_dir --ckpts_tag
python inference_server_simulate.py --model r18-fpn-3x --opt cluster --ckpts_dir --ckpts_tag --tag --budget 10

python inference_server_simulate.py --opt adapt --model r18-fpn-3x --ckpt ../../models/mscoco2017_remap_r18-fpn-3x.pth --smallscale 1 --train_whole 1 --incremental_videos 1 --tag budget10 --budget 10 --iters 600 --eval_interval 301 --save_interval 200 --image_batch_size 2 --num_workers 1 --clustering 1 --buffer_size 4 --update_cluster_interval 205

python inference_server_simulate.py --opt adapt --model r18-fpn-3x --ckpt ../../models/mscoco2017_remap_r18-fpn-3x.pth --smallscale 1 --train_whole 1 --tag budget1 --budget 1 --iters 600 --eval_interval 301 --save_interval 301 --image_batch_size 4 --num_workers 2

python inference_server_simulate.py --opt convert --ckpts_dir . --ckpts_tag adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180
python inference_server_simulate.py --train_whole 1 --opt adapt --model r18-fpn-3x --ckpt_teacher ../../../models/mscoco2017_remap_r18-fpn-3x.pth --ckpt adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.single.pth --tag r18.stage2.10means --resume_prefix r18.stage1.10means --budget 10 --iters 100 --eval_interval 200 --save_interval 200 --image_batch_size 2 --num_workers 2 --outputdir .

python inference_server_simulate.py --model r18-fpn-3x --opt server --ckpts_dir . --ckpts_tag adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.r18.stage2.10means.iter.100

'''
