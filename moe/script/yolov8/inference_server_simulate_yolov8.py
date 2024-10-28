#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
from copy import deepcopy
import gzip
import math
import random
import tqdm
import glob
import psutil
import hashlib
import argparse
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
from multiprocessing import Pool as ProcessPool
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata
import torchvision
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances
from detectron2.config import get_cfg

from yolov8 import *

import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


# Global Variables
video_id_list = [
    '001', '003', '005', '006', '007', '008', '009', '011', '012', '013', 
    '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', 
    '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', 
    '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', 
    '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', 
    '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', 
    '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', 
    '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', 
    '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', 
    '161', '164', '167', '169', '170', '171', '172', '175', '178', '179'
]
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_inference_server')


class YOLOServer(DetectionModel):
    """YOLOServer class that manages training and inference for YOLO models."""

    def get_training_assignment(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Assigns a training path based on the video ID of the inputs."""
        for im in batched_inputs:
            if im['video_id'] != 'coco' and im['video_id'] not in self.video_id_to_index:
                if len(self.un_used_indices) > 0:
                    i = sorted(list(self.un_used_indices.keys()))[0]
                    self.video_id_to_index[im['video_id']] = i
                    del self.un_used_indices[i]
                    self.used_indices[i] = True
                else:
                    self.video_id_to_index[im['video_id']] = np.random.choice(list(self.used_indices.keys()))

        module_indices = []
        for im in batched_inputs:
            # Randomly train 1 path for COCO images
            if im['video_id'] == 'coco':
                if len(self.used_indices) > 0:
                    module_indices.append(np.random.choice(list(self.used_indices.keys())))
                else:
                    module_indices.append(np.random.choice(list(self.un_used_indices.keys())))
            else:
                module_indices.append(self.video_id_to_index[im['video_id']])

        return batched_inputs, module_indices

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], augment=False, profile=False, visualize=False):
        """Performs forward pass for training or inference."""
        if not self.training:
            return self.inference(batched_inputs)

        batched_inputs, module_indices = self.get_training_assignment(batched_inputs)
        batch = self.preprocess_image(batched_inputs)
        x = batch['img']

        # Set module indices for specific layers
        if 9 in self.split_list:
            self.model[9].cv2.module_indices = module_indices
        if 12 in self.split_list:
            self.model[12].m[0].module_indices = module_indices
        if len(self.model) - 1 in self.split_list:
            self.model[-1].module_indices = module_indices

        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        y = []  # outputs
        for i, m in enumerate(self.model):
            if isinstance(m, torch.nn.ModuleList):
                f = m[0].f
                m_idx = m[0].i 
            else:
                f = m.f 
                m_idx = m.i

            if f != -1:  # if not from previous layer
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
            if i in self.split_list:  # split on specific layers
                if type(x) != list and i not in [9, 12]:
                    x = [m[b_id](x[j:j + 1]) for j, b_id in enumerate(module_indices)]
                    x = torch.cat(x, dim=0)
                elif type(x) == list and i != len(self.model) - 1:
                    out = [m[b_id]([xi[j:j + 1] for xi in x]) for j, b_id in enumerate(module_indices)]
                    x = torch.cat(out, dim=0)             
                else:
                    x = m(x)
            else:
                x = m(x)  # run layer
            y.append(x if m_idx in self.save else None)  # save output

        loss, loss_components = self.criterion(x, batch)
        return loss, loss_components

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], augment=False, profile=False, visualize=False, feature=False):
        """Runs inference on the input batch."""
        assert not self.training
        assert 'proposals' not in batched_inputs[0], 'pre-computed proposals not supported'

        module_indices = []
        for im in batched_inputs:
            if im['video_id'] == 'coco' or im['video_id'] not in self.video_id_to_index:
                module_indices.append(0)
            else:
                module_indices.append(self.video_id_to_index[im['video_id']])

        x = self.preprocess_image(batched_inputs)['img']

        # Set module indices for specific layers
        if 9 in self.split_list:
            self.model[9].cv2.module_indices = module_indices
        if 12 in self.split_list:
            self.model[12].m[0].module_indices = module_indices
        if len(self.model) - 1 in self.split_list:
            self.model[-1].module_indices = module_indices

        y, features = [], []

        for i, m in enumerate(self.model):
            if isinstance(m, torch.nn.ModuleList):
                f = m[0].f
                m_idx = m[0].i 
            else:
                f = m.f 
                m_idx = m.i

            if f != -1:  # if not from previous layer
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers

            if i in self.split_list:  # split on specific layers
                if type(x) != list and i not in [9, 12]:
                    x = [m[b_id](x[j:j + 1]) for j, b_id in enumerate(module_indices)]
                    x = torch.cat(x, dim=0)
                elif type(x) == list and i != len(self.model) - 1:
                    out = [m[b_id]([xi[j:j + 1] for xi in x]) for j, b_id in enumerate(module_indices)]
                    x = torch.cat(out, dim=0)             
                else:
                    x = m(x)

            else:
                x = m(x)  # run layer
            y.append(x if m_idx in self.save else None)  # save output

            if (i in [9, 12]) and feature:
                features.append(x)
                if i == 12:
                    return features

        if feature:
            return features

        outputs = non_max_suppression(x[0], conf_thres=0.25, iou_thres=0.45)
        reversed_outputs = self.reverse_yolo_transform(outputs, batched_inputs)
        results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(
            reversed_outputs, batched_inputs, [im['image'].shape[1:] for im in batched_inputs]
        )
        return results

    @classmethod
    def create_from_sup(cls, net, budget, split_list):
        """Creates a YOLOServer instance with specified budget and split list."""
        net.budget = budget
        net.video_id_to_index = {}
        net.used_indices = {}
        net.un_used_indices = {i: True for i in range(0, net.budget)}

        for i in range(len(split_list)):
            if split_list[i] == -1:
                split_list[i] = len(net.model) - 1
        net.split_list = split_list
        
        for i in split_list:
            # Split on specific layers
            if i == 9:
                net.model[i].cv2 = MakeMoE(net.model[i].cv2, budget)
            elif i == 12:
                net.model[i].m[0] = MakeMoE(net.model[i].m[0], budget)
            elif i != len(net.model) - 1:
                net.model[i] = torch.nn.ModuleList([deepcopy(net.model[i]) for _ in range(net.budget)]) 
            else:
                net.model[i] = MoEDetect(net.model[i], budget)

        net.__class__ = cls
        return net


class AdaptativePartialTrainer(DefaultTrainer):
    """Trainer class for adaptive partial training of YOLO models."""

    def __init__(self, cfg, train_whole, incremental_videos, ckpt):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):
            detectron2.utils.logger.setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        model = load_yolov8(os.path.join(os.path.dirname(__file__), cfg.YOLO_CONFIG_PATH), ckpt)

        self.model_teacher = load_yolov8(os.path.join(os.path.dirname(__file__), cfg.YOLO_CONFIG_PATH), "../../models/yolov8s_remap.pth")
        self.model_teacher.output_format = "yolo"
        self.model_teacher.eval()
 
        model = YOLOServer.create_from_sup(model, cfg.MODEL.ADAPTIVE_BUDGET, cfg.MODEL.SPLIT_LIST)

        if cfg.MODEL.MAPPER is not None:
            mapper = torch.load(cfg.MODEL.MAPPER)
            model.budget = mapper['budget']
            model.video_id_to_index = mapper['video_id_to_index']
            model.used_indices = mapper['used_indices']
            model.un_used_indices = mapper['un_used_indices']

        if train_whole:
            trainable_modules = [model]
        else:
            trainable_modules = [model.model[i] for i in cfg.MODEL.SPLIT_LIST]

        _count_all, _count_train = 0, 0
        for p in model.parameters():
            _count_all += p.numel()
            p.requires_grad = False
        for m in trainable_modules:
            for p in m.parameters():
                _count_train += p.numel()
                p.requires_grad = True

        optimizer = self.build_optimizer(cfg, torch.nn.ModuleList(trainable_modules))
        if incremental_videos:
            data_loader = detectron2.data.build_detection_train_loader(
                cfg, sampler=detectron2.data.samplers.distributed_sampler.TrainingSampler(cfg.DATASETS.TRAIN_NUM_IMAGES, shuffle=False)
            )
        else:
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
        """Builds hooks for evaluation and saving model states."""
        ret = super().build_hooks()
        self.eval_results_all = {}

        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = deepcopy(self._last_eval_results)
            return self._last_eval_results

        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)

        def save_model_state():
            model = self.model
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            prefix = '%s.iter.%d' % (self.cfg.SOLVER.SAVE_PREFIX, self.iter)
            torch.save(model.state_dict(), prefix + '.pth')
            mapper = {
                'budget': model.budget,
                'video_id_to_index': model.video_id_to_index,
                'used_indices': model.used_indices,
                'un_used_indices': model.un_used_indices
            }
            torch.save(mapper, prefix + '.mapper.pth')

        ret.append(detectron2.engine.hooks.EvalHook(self.cfg.SOLVER.SAVE_INTERVAL, save_model_state))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Builds evaluator for COCO format."""
        return COCOEvaluator(dataset_name, output_dir=finetune_output)


def finetune_ema_simple_trainer_run_step(self):
    """Defines the step to run during training for fine-tuning."""
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
            pseudo_labels = self.model_teacher.inference(pseudo_inputs)
            for _i, _pred in zip(pseudo_idx, pseudo_labels):
                data[_i]['instances'] = _pred
                del data[_i]['image_test']

    loss, loss_components = self.model(data)

    loss_dict_items = {
        "Box loss": float(loss_components[0]),
        "Class loss": float(loss_components[1]),
        "DFL loss": float(loss_components[2])
    }
    loss_dict = {
        "Box loss": loss_components[0],
        "Class loss": loss_components[1],
        "DFL loss": loss_components[2]
    }

    self.optimizer.zero_grad()
    loss.backward()
    self._write_metrics(loss_dict, data_time)
    self.optimizer.step()

    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


class DatasetMapperPseudo(detectron2.data.DatasetMapper):
    """DatasetMapper class for generating pseudo-labels during training."""

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = deepcopy(dataset_dict)  # it will be modified by code below
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)

        if 'source' in dataset_dict and dataset_dict['source'] == 'unlabeled':
            image_test = self.apply_test_transform(image)
            if image_test is not None:
                dataset_dict['image_test'] = torch.as_tensor(np.ascontiguousarray(image_test.transpose(2, 0, 1)))

        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    def apply_test_transform(self, image):
        """Applies test-time transformations to the input image."""
        if not (image.dtype == np.uint8 and len(image.shape) == 3 and image.shape[2] == 3):
            return None
        h, w = image.shape[:2]
        scale = 2
        min_size, max_size = map(lambda x: int(x * scale), [self.min_size_test, self.max_size_test])
        newh, neww = self.get_output_shape(h, w, min_size, max_size)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
        return np.asarray(pil_image)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
        """Calculates new output dimensions for the image."""
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
        """Creates a DatasetMapperPseudo instance from a base mapper."""
        assert not cfg.INPUT.CROP.ENABLED
        assert cfg.INPUT.RANDOM_FLIP == 'none'
        mapper.min_size_test, mapper.max_size_test = cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperPseudo
        return mapper


def adapt(args):
    """Adapts the model and prepares datasets based on provided arguments."""
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    random.seed(42)

    # Load validation images
    desc_manual_valid, dst_manual_valid = 'allvideos_manual', []
    for v in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'annotation', v)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        for i in range(len(annotations)):
            annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
            annotations[i]['video_id'] = v
        print(f'manual annotation for {v}: {len(annotations)} images, {sum(len(a["annotations"]) for a in annotations)} bboxes')
        dst_manual_valid.extend(annotations)
    for i, ann in enumerate(dst_manual_valid):
        assert 'video_id' in ann
        ann['image_id'] = i + 1
    print(f'manual annotation for all videos: {len(dst_manual_valid)} images, {sum(len(a["annotations"]) for a in dst_manual_valid)} bboxes')

    # Load training images
    dst_cocotrain = get_coco_dicts(args, 'train')
    for im in dst_cocotrain:
        im['source'] = 'coco'
        im['video_id'] = 'coco'
    random.shuffle(dst_cocotrain)

    # Prepare pseudo-annotations
    dst_pseudo_anno, desc_pseudo_anno = [], 'allvideos_unlabeled_cocotrain'
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    for v in video_id_list:
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'images', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = meta['ifilelist']
        dict_json = [{
            'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', fname)),
            'image_id': i,
            'height': meta['meta']['video']['H'],
            'width': meta['meta']['video']['W'],
            'annotations': [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}],
            'source': 'unlabeled',
            'video_id': v
        } for i, fname in enumerate(ifilelist)]
        
        print(f'unlabeled frames of video {v} at {lmdb_path}: {len(dict_json)} images')
        if len(dict_json) > images_per_video_cap:
            random.shuffle(dict_json)
            dict_json = dict_json[:images_per_video_cap]
            print(f'randomly downsampled to: {len(dict_json)} images')
        dst_pseudo_anno.extend(dict_json)
    print(f'total unlabeled: {len(dst_pseudo_anno)} images')

    if args.incremental_videos:
        # Ensure the number of training frames matches iterations * image-batchsize
        while len(dst_pseudo_anno) < args.iters * args.image_batch_size:
            dst_pseudo_anno.extend(dst_pseudo_anno)
        random.shuffle(dst_pseudo_anno)
        dst_pseudo_anno = dst_pseudo_anno[: args.iters * args.image_batch_size * 3 // 4]
        dst_pseudo_anno.sort(key=lambda x: hashlib.md5(x['video_id'].encode('utf-8')).hexdigest() + os.path.basename(x['file_name']))
        dst_pseudo_anno_with_coco = []
        for i in range(0, len(dst_pseudo_anno) // 3 - 1):
            dst_pseudo_anno_with_coco.extend(dst_pseudo_anno[i * 3:(i + 1) * 3])
            dst_pseudo_anno_with_coco.append(dst_cocotrain[i % len(dst_cocotrain)])
        while len(dst_pseudo_anno_with_coco) < (args.iters + 2) * args.image_batch_size:
            dst_pseudo_anno_with_coco.append(deepcopy(dst_pseudo_anno_with_coco[-1]))
        dst_pseudo_anno = dst_pseudo_anno_with_coco
        assert len(dst_pseudo_anno) >= (args.iters + 2) * args.image_batch_size
    else:
        # Combine pseudo-annotations with MSCOCO
        dst_pseudo_anno.extend(dst_cocotrain[: len(dst_pseudo_anno) // 3])
    
    for i, ann in enumerate(dst_pseudo_anno):
        assert 'video_id' in ann
        ann['image_id'] = i + 1
    print(f'include MSCOCO2017 training images, totally {len(dst_pseudo_anno)} images')
    prefix = f'adaptive_partial_server_yolov8s_anno_{desc_pseudo_anno}.{args.tag}'

    # Clean up and register datasets
    del _tensor
    gc.collect()
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output

    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 200
    cfg.SOLVER.GAMMA = 1
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.PSEUDO_DET_MIN_SCORE = args.refine_det_score_thres
    cfg.SOLVER.SAVE_INTERVAL = args.save_interval
    cfg.SOLVER.SAVE_PREFIX = os.path.join(args.outputdir, prefix)

    cfg.TEST.EVAL_PERIOD = args.eval_interval
    
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid,)
    cfg.DATASETS.TRAIN_NUM_IMAGES = len(dst_pseudo_anno)
    cfg.DATASETS.TEST_NUM_IMAGES = len(dst_manual_valid)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    cfg.MODEL.SPLIT_LIST = args.split_list
    cfg.MODEL.MAPPER = args.mapper

    cfg.YOLO_CONFIG_PATH = args.config

    print(cfg)

    # Adjust training parameters
    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = AdaptativePartialTrainer(cfg, args.train_whole, args.incremental_videos, args.ckpt)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_ema_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperPseudo.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, cfg)

    # Manually load the parameters from checkpoint
    print(f'loading weights from: {args.ckpt}')
    assert trainer.model is trainer._trainer.model
    assert trainer.model_teacher is trainer._trainer.model_teacher

    print(prefix)
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print(f'Evaluate on {dataset_name}')
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(args.outputdir, f'{prefix}.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)

    # Plotting results
    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(aps.keys())
    dst_list = {'mAP': [], 'AP50': []}
    for i in iter_list:
        dst_list['mAP'].append(aps[i]['bbox']['AP'])
        dst_list['AP50'].append(aps[i]['bbox']['AP50'])

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for x in loss_history:
        for loss_key, loss_val in x['loss'].items():
            if loss_key not in loss_history_dict:
                loss_history_dict[loss_key] = []
            loss_history_dict[loss_key].append([x['iter'], loss_val])
    loss_history_dict = {loss_key: np.array(vals) for loss_key, vals in loss_history_dict.items()}
    for loss_key, loss_vals in loss_history_dict.items():
        for i in range(smooth_L, loss_vals.shape[0]):
            loss_vals[i, 1] = loss_vals[i - smooth_L:i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_vals[smooth_L + 1:, :]

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
    for loss_key, loss_vals in loss_history_dict.items():
        plt.plot(loss_vals[:, 0], loss_vals[:, 1], linestyle='-', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, f'{prefix}.pdf'))


class SemiRandomClient(torchdata.Dataset):
    """Dataset class for semi-random client data loading."""

    def __init__(self, cfg):
        super(SemiRandomClient, self).__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == 'BGR'

        self.images = []
        for video_id in video_id_list:
            inputdir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'annotation', video_id)
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                _dicts = json.load(fp)
            for im in _dicts:
                im['md5'] = hashlib.md5(f"{video_id}_{im['file_name']}".encode('utf-8')).hexdigest()
                im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))
                im['video_id'] = video_id
            self.images.extend(_dicts)
        self.images.sort(key=lambda x: x['md5'])
        self.preloaded_images = None

    def preload(self):
        """Preloads images into memory."""
        if self.preloaded_images is not None:
            return
        self.preloaded_images = []
        for i in tqdm.tqdm(range(len(self.images)), ascii=True, desc='preloading images'):
            self.preloaded_images.append(self.read(i))

    def __len__(self):
        return len(self.images)

    def read(self, i):
        """Reads an image and applies transformations."""
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


class _MapperDataset(torchdata.Dataset):
    """Dataset class for mapping images."""

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


def random_cluster(args):
    """Performs random clustering and saves the mapper."""
    for seed in args.gen_seed:
        random.seed(seed)
        mapper = {'budget': args.budget, 'video_id_to_index': {}, 'used_indices': {}, 'un_used_indices': {b: True for b in range(args.budget)}}
        index_list = list(range(args.budget))
        for idx in video_id_list:
            choice = random.choice(index_list)
            mapper['video_id_to_index'][idx] = choice
            mapper['used_indices'][choice] = True
            if choice in mapper['un_used_indices']:
                del mapper['un_used_indices'][choice]
        print(mapper)
        torch.save(mapper, os.path.join(args.outputdir, f"mapper_random_{seed}_b{args.budget}.pth"))


def offline_cluster(args):
    """Performs offline clustering using K-Means and saves the mapper."""
    from sklearn.cluster import KMeans

    if args.from_base:
        model = load_yolov8(args.config, args.ckpt)
        model = YOLOServer.create_from_sup(model, 1, args.split_list)
    else:
        # Load trained B=1 model
        mapper = torch.load(os.path.join(args.ckpts_dir, f'{args.ckpts_tag}.mapper.pth'))
        assert mapper['budget'] == 1, 'can only run offline clustering from budget=1 model'
        
        model = load_yolov8(args.config)
        model = YOLOServer.create_from_sup(model, mapper['budget'], args.split_list)
        state_dict = torch.load(os.path.join(args.ckpts_dir, f'{args.ckpts_tag}.pth'))
        model.load_state_dict(state_dict)
        print(f'loaded weights from: {os.path.join(args.ckpts_dir, f"{args.ckpts_tag}.pth")}')
        model.budget = mapper['budget']
        model.video_id_to_index = mapper['video_id_to_index']
        model.used_indices = mapper['used_indices']
        model.un_used_indices = mapper['un_used_indices']
    model.eval()
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    video_images, num_images_per_video = [], 400
    for v in tqdm.tqdm(video_id_list, ascii=True, desc='loading images'):
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'images', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = sorted(meta['ifilelist'])
        assert len(ifilelist) / num_images_per_video > 2
        ifilelist = np.array(ifilelist[: (len(ifilelist) // num_images_per_video) * num_images_per_video])
        ifilelist = ifilelist.reshape(-1, num_images_per_video)[0]
        for i, fname in enumerate(ifilelist):
            video_images.append({
                'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', fname)),
                'image_id': i,
                'height': meta['meta']['video']['H'],
                'width': meta['meta']['video']['W'],
                'annotations': [],
                'video_id': v
            })
    print(f'total images: {len(video_images)}')
    dataset = _MapperDataset(cfg, video_images)
    loader = torchdata.DataLoader(dataset, batch_size=args.image_batch_size, collate_fn=_MapperDataset.collate, shuffle=False, num_workers=6)
    
    video_id_features = []
    video_features_p3, video_features_p4, video_features_p5 = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, ascii=True, desc='extracting features'):
            video_id_features.extend([im['video_id'] for im in batch])
            features = model.inference(batch, feature=True)
            features_p4 = torch.nn.functional.adaptive_avg_pool2d(features[1], (9, 16)).view(len(batch), -1)
            video_features_p4.append(features_p4.detach().cpu())
            features_p5 = torch.nn.functional.adaptive_avg_pool2d(features[0], (5, 9)).view(len(batch), -1)
            video_features_p5.append(features_p5.detach().cpu())
    
    video_id_features = np.array(video_id_features)
    video_features_p4 = torch.cat(video_features_p4, dim=0).detach().numpy()
    video_features_p5 = torch.cat(video_features_p5, dim=0).detach().numpy()
    torch.cuda.empty_cache()
    
    for features, desc in [
            (video_features_p4, 'fpn.p4'),
            (video_features_p5, 'fpn.p5'),
            (np.concatenate([video_features_p4, video_features_p5], axis=1), 'fpn.p4.p5')]:
        
        print(f'running {args.budget}-Means for {desc}: {features.shape} {features.dtype}')
        kmeans = KMeans(n_clusters=args.budget, random_state=0).fit(features)
        mapper = {'budget': args.budget, 'video_id_to_index': {}, 'used_indices': {}, 'un_used_indices': {b: True for b in range(args.budget)}}
        for v in video_id_list:
            cluster_ids = kmeans.labels_[video_id_features == v]
            i = np.argmax(np.bincount(cluster_ids))
            mapper['video_id_to_index'][v] = i
            mapper['used_indices'][i] = True
            if i in mapper['un_used_indices']:
                del mapper['un_used_indices'][i]
        print(mapper)
        prefix = os.path.join(args.ckpts_dir, f'{args.ckpts_tag}.{args.budget}means.{desc}')
        torch.save(mapper, prefix + f'.new{".frombase" if args.from_base else ""}.mapper.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, default='server', help='option')
    parser.add_argument('--config', type=str, default='../../configs/yolov8s.yaml', help='detection model config path')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default="../../models/yolov8s_remap.pth", help='weights checkpoint of model')
    parser.add_argument('--mapper', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--cocodir', type=str, default='../../../mscoco')
    parser.add_argument('--smallscale', type=bool, default=False)
    parser.add_argument('--incremental_videos', type=bool, default=False)
    parser.add_argument('--train_whole', type=bool, default=False)

    # cluster parameters
    parser.add_argument('--split_list', type=int, nargs='+')
    parser.add_argument('--from_base', type=bool, default=False)
    parser.add_argument('--random', type=bool, default=False)
    parser.add_argument('--gen_seed', type=int, nargs='+')

    parser.add_argument('--ckpts_dir', type=str, default=None, help='weights checkpoints of individual models')
    parser.add_argument('--ckpts_tag', type=str, default='')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)
    parser.add_argument('--preload', type=bool, default=False)
    parser.add_argument('--instances', type=int, default=1)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--save_interval', type=int, help='interval for saving model')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.00334, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)

    # used for random test
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.id != '':
        video_id_list = [args.id]

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    finetune_output = args.outputdir
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'adapt':
        if args.ddp_num_gpus <= 1:
            adapt(args)
        else:
            from detectron2.engine import launch
            launch(adapt, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    if args.opt == 'cluster':
        if args.random:
            random_cluster(args)
        else:
            offline_cluster(args)

