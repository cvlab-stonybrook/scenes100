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
import itertools
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
from detectron2.modeling import detector_postprocess, build_model
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model, default_setup
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances
from detectron2.config import get_cfg
from dino import *
from config import add_dino_config
from util.get_param_dicts import get_param_dict
from cfg_to_args import cfg_to_args 

import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_detector_train import get_coco_dicts


# Global variables
video_id_list = [
    '001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', 
    '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', 
    '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', 
    '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', 
    '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', 
    '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', 
    '161', '164', '167', '169', '170', '171', '172', '175', '178', '179'
]

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_inference_server')


class MakeMoE(torch.nn.Module):
    """
    MoE model using multiple experts for a given budget.
    """
    def __init__(self, net, budget):
        super(MakeMoE, self).__init__()
        self.experts = torch.nn.ModuleList([deepcopy(net) for _ in range(budget)])

    def forward(self, x):
        assert len(x) == len(self.module_indices) or x.shape[1] == len(self.module_indices), "Invalid input to MoE."
        if len(x) == len(self.module_indices):
            out = [self.experts[m](x[i:i+1, :]) for i, m in enumerate(self.module_indices)]
            out = torch.cat(out, dim=0)
        else:
            out = [self.experts[m](x[:, i:i+1, :]) for i, m in enumerate(self.module_indices)]
            out = torch.cat(out, dim=1)
        return out


class DINOServer(Dino):
    """
    DINO Server model for handling video-specific training assignments and inference.
    """
    def get_training_assignment(self, batched_inputs):
        module_indices = []
        if self.training:
            for im in batched_inputs:
                if im['video_id'] != 'coco' and im['video_id'] not in self.video_id_to_index:
                    if self.un_used_indices:
                        i = sorted(self.un_used_indices.keys())[0]
                        self.video_id_to_index[im['video_id']] = i
                        del self.un_used_indices[i]
                        self.used_indices[i] = True
                    else:
                        self.video_id_to_index[im['video_id']] = np.random.choice(list(self.used_indices.keys()))
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
        else:
            module_indices = []
            for im in batched_inputs:
                # at inference time, assign first module for all unseen video IDs
                if 'video_id' not in im:
                    module_indices.append(0)
                    continue
                if im['video_id'] == 'coco' or im['video_id'] not in self.video_id_to_index:
                    module_indices.append(0)
                else:
                    module_indices.append(self.video_id_to_index[im['video_id']])
        return batched_inputs, module_indices

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
        batched_inputs, module_indices = self.get_training_assignment(batched_inputs)

        # Assign module indices to submodules
        if not self.interm:
            self.model.backbone[0].body.conv1.module_indices = module_indices
            self.model.backbone[0].body.bn1.module_indices = module_indices
            self.model.backbone[0].body.layer1.module_indices = module_indices
            self.model.bbox_embed[0].module_indices = module_indices
            self.model.class_embed[0].module_indices = module_indices
            self.model.transformer.enc_out_bbox_embed.module_indices = module_indices
            self.model.transformer.enc_out_class_embed.module_indices = module_indices
        else:
            self.model.backbone[0].body.layer3[0].conv1.module_indices = module_indices
            self.model.backbone[0].body.layer3[0].conv2.module_indices = module_indices

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
            for k in loss_dict.keys():
                if k in self.criterion.weight_dict:
                    loss_dict[k] *= self.criterion.weight_dict[k]
            return loss_dict
        else:
            box_cls, box_pred = output["pred_logits"], output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []

            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
                ratio = input_per_image['image'].shape[1] / height
                # self.draw(input_per_image['image']/255, r, ratio=ratio, file_name=f'test.png')
            return processed_results

    @classmethod
    def create_from_sup(cls, net, budget, interm=False):
        """
        Initialize MoE architecture for DINO Server with specified budget and query ratios.
        """
        net.budget = budget
        print(f'Module budget: {budget}')
        net.video_id_to_index = {}
        net.used_indices = {}
        net.un_used_indices = {i: True for i in range(net.budget)}

        # Convert B=1 model to base model
        # net.model.backbone[0].body.conv1 = net.model.backbone[0].body.conv1.experts[0]
        # net.model.backbone[0].body.bn1 = net.model.backbone[0].body.bn1.experts[0]
        # net.model.backbone[0].body.layer1 = net.model.backbone[0].body.layer1.experts[0]
        # net.model.transformer.decoder.bbox_embed[0] = net.model.transformer.decoder.bbox_embed[0].experts[0]
        # net.model.transformer.decoder.class_embed[0] = net.model.transformer.decoder.class_embed[0].experts[0]
        # net.model.transformer.enc_out_bbox_embed = net.model.transformer.enc_out_bbox_embed.experts[0]
        # net.model.transformer.enc_out_class_embed = net.model.transformer.enc_out_class_embed.experts[0]
        # for i in range(len(net.model.transformer.decoder.layers)):
        #     net.model.transformer.decoder.bbox_embed[i] = net.model.transformer.decoder.bbox_embed[0]
        #     net.model.transformer.decoder.class_embed[i] = net.model.transformer.decoder.class_embed[0]
        # net.model.bbox_embed = net.model.transformer.decoder.bbox_embed
        # net.model.class_embed = net.model.transformer.decoder.class_embed               

        if not interm:
            # MoE assignments
            net.model.backbone[0].body.conv1 = MakeMoE(net.model.backbone[0].body.conv1, budget)
            net.model.backbone[0].body.bn1 = MakeMoE(net.model.backbone[0].body.bn1, budget)
            net.model.backbone[0].body.layer1 = MakeMoE(net.model.backbone[0].body.layer1, budget)
            net.model.transformer.decoder.bbox_embed[0] = MakeMoE(net.model.transformer.decoder.bbox_embed[0], budget)
            net.model.transformer.decoder.class_embed[0] = MakeMoE(net.model.transformer.decoder.class_embed[0], budget)
            for i in range(len(net.model.transformer.decoder.layers)):
                net.model.transformer.decoder.bbox_embed[i] = net.model.transformer.decoder.bbox_embed[0]
                net.model.transformer.decoder.class_embed[i] = net.model.transformer.decoder.class_embed[0]
            net.model.transformer.enc_out_bbox_embed = MakeMoE(net.model.transformer.enc_out_bbox_embed, budget)
            net.model.transformer.enc_out_class_embed = MakeMoE(net.model.transformer.enc_out_class_embed, budget)
            net.model.bbox_embed = net.model.transformer.decoder.bbox_embed
            net.model.class_embed = net.model.transformer.decoder.class_embed
        else:
            # Optional Intermediate Split
            net.model.backbone[0].body.layer3[0].conv1 = MakeMoE(net.model.backbone[0].body.layer3[0].conv1, budget)
            net.model.backbone[0].body.layer3[0].conv2 = MakeMoE(net.model.backbone[0].body.layer3[0].conv2, budget)
        
        net.interm = interm
        net.__class__ = cls
        return net


class AdaptativePartialTrainer(DefaultTrainer):
    """
    Trainer for Adaptive Partial Finetuning of DINO models.
    """
    def __init__(self, cfg, incremental_videos, ckpt):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # Setup logger if not initialized
            detectron2.utils.logger.setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())
        model = self.build_model(cfg)

        try:
            state_dict = torch.load(ckpt)['model']
        except KeyError:
            state_dict = torch.load(ckpt)
        
        model.load_state_dict(state_dict, strict=False)

        if cfg.MODEL.MAPPER is not None:
            mapper = torch.load(cfg.MODEL.MAPPER)
            model.budget = mapper['budget']
            model.video_id_to_index = mapper['video_id_to_index']
            model.used_indices = mapper['used_indices']
            model.un_used_indices = mapper['un_used_indices']

        self.model_teacher = self.build_model(cfg)
        self.model_teacher.load_state_dict(torch.load("../../models/dino_5scale_remap_orig.pth")['model'])
        self.model_teacher.eval()

        model = DINOServer.create_from_sup(model, cfg.MODEL.ADAPTIVE_BUDGET, cfg.MODEL.INTERM)
        trainable_modules = [model]
        _count_all, _count_train = 0, 0
        for p in model.parameters():
            _count_all += p.numel()
            p.requires_grad = False
        for m in trainable_modules:
            try:
                for p in m.parameters():
                    _count_train += p.numel()
                    p.requires_grad = True
            except:
                _count_train += m.numel()
                m.requires_grad = True
        print('Only train subset of model parameters: %d/%d %.4f%%' % (_count_train, _count_all, _count_train / _count_all * 100))
        try:
            optimizer = self.build_optimizer(cfg, torch.nn.ModuleList(trainable_modules))
        except:
            optimizer = torch.optim.Adam(trainable_modules, lr=cfg.SOLVER.BASE_LR)

        if incremental_videos:
            data_loader = detectron2.data.build_detection_train_loader(
                cfg,
                sampler=detectron2.data.samplers.distributed_sampler.TrainingSampler(cfg.DATASETS.TRAIN_NUM_IMAGES, shuffle=False)
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
        """
        Build hooks for evaluation and model state saving.
        """
        ret = super().build_hooks()
        self.eval_results_all = {}

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = deepcopy(self._last_eval_results)
            return self._last_eval_results

        # Replace EvalHook with the custom function
        for i, hook in enumerate(ret):
            if isinstance(hook, detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results)

        def save_model_state():
            model = self.model
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                print('Unwrapping data parallel')
                model = model.module

            prefix = f'{self.cfg.SOLVER.SAVE_PREFIX}.iter.{self.iter}'
            torch.save(model.state_dict(), f'{prefix}.pth')
            mapper = {
                'budget': model.budget,
                'video_id_to_index': model.video_id_to_index,
                'used_indices': model.used_indices,
                'un_used_indices': model.un_used_indices
            }
            torch.save(mapper, f'{prefix}.mapper.pth')
            print(f'Saved model state to: {prefix}')

        ret.append(detectron2.engine.hooks.EvalHook(self.cfg.SOLVER.SAVE_INTERVAL, save_model_state))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=finetune_output)

    @classmethod
    def build_optimizer(cls, cfg, model):
        plain_model = model[0].model
        args = cfg_to_args(cfg)
        param_dicts = get_param_dict(args, plain_model)
        
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.BASE_LR,
                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return optimizer


def finetune_ema_simple_trainer_run_step(self):
    """
    Run training step.
    """
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start

    pseudo_idx, pseudo_inputs = [], []
    for idx, d in enumerate(data):
        if 'image_test' in d:
            pseudo_idx.append(idx)
            height, width = d['instances'].image_size
            pseudo_inputs.append({'image': d['image_test'], 'height': height, 'width': width})

    if pseudo_idx:
        with torch.no_grad():
            pseudo_labels = [self.model_teacher.inference_split(inputs) for inputs in pseudo_inputs]
            for idx, pred in zip(pseudo_idx, pseudo_labels):
                mask = pred['instances'].scores >= self.pseudo_det_min_score
                filtered = Instances(pred['instances']._image_size)
                filtered.set('gt_boxes', pred['instances'].pred_boxes[mask])
                filtered.set('gt_classes', pred['instances'].pred_classes[mask])
                data[idx]['instances'] = filtered
                del data[idx]['image_test']

    # import matplotlib.patches as patches
    # _, axes = plt.subplots(1, len(data)); axes = axes.reshape(-1)
    # for _i, _d in enumerate(data):
    #     print(_d['image'].size(), _d['instances']._image_size)
    #     _im = _d['image'][0].detach().cpu().numpy(); axes[_i].imshow(_im)
    #     for j in range(0, len(_d['instances'])):
    #         x1, y1, x2, y2 = _d['instances'].gt_boxes.tensor[j].detach().cpu().numpy()
    #         k = _d['instances'].gt_classes[j].detach().cpu().numpy()
    #         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
    #         axes[_i].add_patch(rect)
    # plt.show()

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
        dataset_dict = deepcopy(dataset_dict)  # it will be modified by code below
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
        # scale = np.random.rand() * 0.75 + 1.5 # upscale by 1.5 ~ 2.25
        scale = 1
        min_size, max_size = map(lambda x: int(x * scale), [self.min_size_test, self.max_size_test])
        newh, neww = self.get_output_shape(h, w, min_size, max_size)
        pil_image = Image.fromarray(image)
        # pil_image = pil_image.resize((neww, newh), Image.Resampling.BILINEAR)
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
        mapper.min_size_test, mapper.max_size_test = cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperPseudo
        return mapper


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    random.seed(42)

    # validation images
    desc_manual_valid, dst_manual_valid = 'allvideos_manual', []
    for v in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', v)
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
    dst_cocotrain = get_coco_dicts(args, 'train')
    for im in dst_cocotrain:
        im['source'] = 'coco'
        im['video_id'] = 'coco'
    random.shuffle(dst_cocotrain)

    dst_pseudo_anno, desc_pseudo_anno = [], 'allvideos_unlabeled_cocotrain'
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    for v in video_id_list:
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v))
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

    if args.incremental_videos:
        # for video incremental training, make sure (# of training frames) == iterations * image-batchsize
        # 1/4 of images are from MSCOCO
        while len(dst_pseudo_anno) < args.iters * args.image_batch_size:
            dst_pseudo_anno = dst_pseudo_anno + dst_pseudo_anno
        random.shuffle(dst_pseudo_anno)
        dst_pseudo_anno = dst_pseudo_anno[: args.iters * args.image_batch_size * 3 // 4]
        dst_pseudo_anno.sort(key=lambda x: hashlib.md5(x['video_id'].encode('utf-8')).hexdigest() + os.path.basename(x['file_name']))
        dst_pseudo_anno_with_coco = []
        for i in range(0, len(dst_pseudo_anno) // 3 - 1):
            dst_pseudo_anno_with_coco.extend(dst_pseudo_anno[i * 3 : (i + 1) * 3])
            dst_pseudo_anno_with_coco.append(dst_cocotrain[i % len(dst_cocotrain)])
        while len(dst_pseudo_anno_with_coco) < (args.iters + 2) * args.image_batch_size:
            dst_pseudo_anno_with_coco.append(deepcopy(dst_pseudo_anno_with_coco[-1]))
        dst_pseudo_anno = dst_pseudo_anno_with_coco
        del dst_pseudo_anno_with_coco
        assert len(dst_pseudo_anno) >= (args.iters + 2) * args.image_batch_size
    else:
        # 1/4 of images are from MSCOCO
        dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[: len(dst_pseudo_anno) // 3]
    for i in range(0, len(dst_pseudo_anno)):
        assert 'video_id' in dst_pseudo_anno[i]
        dst_pseudo_anno[i]['image_id'] = i + 1
    print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
    prefix = 'adaptive_partial_server_%s_anno_%s.%s' % ("dino", desc_pseudo_anno, args.tag)

    del _tensor
    gc.collect()
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    cfg = get_cfg()
    add_dino_config(cfg)
    default_setup(cfg, args)
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output

    cfg.SOLVER.WARMUP_ITERS = args.iters // 200
    cfg.SOLVER.GAMMA = 1 #
    cfg.SOLVER.STEPS = () #
    cfg.SOLVER.MAX_ITER = args.iters

    cfg.TEST.EVAL_PERIOD = args.eval_interval


    cfg.SOLVER.PSEUDO_DET_MIN_SCORE = args.refine_det_score_thres
    cfg.SOLVER.SAVE_INTERVAL = args.save_interval
    cfg.SOLVER.SAVE_PREFIX = os.path.join(args.outputdir, prefix)
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    cfg.MODEL.MAPPER = args.mapper
    cfg.MODEL.INTERM = args.interm

    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid,)
    cfg.DATASETS.TRAIN_NUM_IMAGES = len(dst_pseudo_anno)
    cfg.DATASETS.TEST_NUM_IMAGES = len(dst_manual_valid)
    
    cfg.CONFIG_PATH = args.config

    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = AdaptativePartialTrainer(cfg, args.incremental_videos, args.ckpt)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_ema_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperPseudo.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, cfg)

    # manually load the parameters from ckpt
    print('loading weights from:', args.ckpt)
    # trainer.resume_or_load(resume=False)
    assert trainer.model is trainer._trainer.model
    assert trainer.model_teacher is trainer._trainer.model_teacher


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
        if color_i <= 6: # show only maximum 6 loss items 
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


class SemiRandomClient(torchdata.Dataset):
    def __init__(self, cfg):
        super(SemiRandomClient, self).__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == 'BGR'

        self.images = []
        for video_id in video_id_list:
            inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
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


def inference_throughput(args):
    mapper = torch.load(os.path.join(args.ckpts_dir, '%s.mapper.pth' % args.ckpts_tag))
    print(mapper)
    cfg = get_cfg_base_model("r101-fpn-3x")
    model = YOLOServer.create_from_sup(load_model(os.path.join(os.path.dirname(__file__),'../../configs/yolov3-custom.cfg')), mapper['budget'])
    model.video_id_to_index = mapper['video_id_to_index']
    model.used_indices = mapper['used_indices']
    model.un_used_indices = mapper['un_used_indices']
    state_dict = torch.load(os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    model.load_state_dict(state_dict)
    del state_dict
    dataset = SemiRandomClient(cfg)
    dataset.images = list(filter(lambda x: x['video_id'] == args.id, dataset.images))
    dataset.images = sorted(dataset.images, key=lambda x: x['file_name'])[:10]
    dataset.preload()
    gc.collect()
    torch.cuda.empty_cache()
    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            model.inference([dataset[i % len(dataset)][0]])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


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


def random_cluster(args):
    for seed in args.gen_seed:
        random.seed(seed)
        mapper = {'budget': args.budget, 'video_id_to_index': {}, 'used_indices': {}, 'un_used_indices': {b: True for b in range(0, args.budget)}}
        index_list = [i for i in range(args.budget)]
        # video_id_to_index = {}
        for idx in video_id_list:
            choice = random.choice(index_list)
            mapper['video_id_to_index'][idx] = choice
            mapper['used_indices'][choice] = True
            if choice in mapper['un_used_indices']:
                del mapper['un_used_indices'][choice]
        print(mapper)
        torch.save(mapper, os.path.join(args.outputdir, f"mapper_random_{seed}_b{args.budget}.pth"))


def offline_cluster(args):
    from sklearn.cluster import KMeans

    cfg = get_cfg()
    add_dino_config(cfg)
    model = build_model(cfg)
    model = DINOServer.create_from_sup(model, args.budget)
    try:
        state_dict = torch.load(args.ckpt)['model']
    except:
        state_dict = torch.load(args.ckpt)
    # model.load_state_dict(convert_base_to_moe(state_dict, cfg), strict=True)
    model.eval()
    video_images, num_images_per_video = [], 400
    for v in tqdm.tqdm(video_id_list, ascii=True, desc='loading images'):
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v))
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
    loader = torchdata.DataLoader(dataset, batch_size=args.image_batch_size, collate_fn=_MapperDataset.collate, shuffle=False, num_workers=6)   
    video_id_features = []
    video_features_p3, video_features_p4, video_features_p5 = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, ascii=True, desc='extracting features'):
            video_id_features.extend([im['video_id'] for im in batch])           
            features = model.forward(batch, return_features=True)
            # image level features
            # features_p3 = torch.nn.functional.adaptive_avg_pool2d(features[0], (12, 21)).view(len(batch), -1)
            # video_features_p3.append(features_p3.detach().cpu())
            # features_p4 = torch.nn.functional.adaptive_avg_pool2d(features[-2], (9, 16)).view(len(batch), -1)
            # video_features_p4.append(features_p4.detach().cpu())
            features_p5 = torch.nn.functional.adaptive_avg_pool2d(features[-1], (5, 9)).view(len(batch), -1)
            video_features_p5.append(features_p5.detach().cpu())
    video_id_features = np.array(video_id_features)
    # video_features_p3 = torch.cat(video_features_p3, dim=0).detach().numpy()
    # video_features_p4 = torch.cat(video_features_p4, dim=0).detach().numpy()
    video_features_p5 = torch.cat(video_features_p5, dim=0).detach().numpy()      
    torch.cuda.empty_cache()
    for features, desc in [
            # (video_features_p3, 'fpn.p3'),
            # (video_features_p4, 'fpn.p4'),
            # (np.concatenate([video_features_p3, video_features_p4], axis=1), 'fpn.p3.p4'),
            (video_features_p5, 'fpn.p5'),
            # (np.concatenate([video_features_p4, video_features_p5], axis=1), 'fpn.p4.p5')
            ]:
        
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
        # torch.save(state_dict, prefix + '.pth')
        torch.save(mapper, prefix + '.mapper.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, default='adapt', help='option')
    parser.add_argument('--config', type=str, default='../../configs/yolov5l.yaml', help='detection model config path')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.3, help='minimum detection score in pseudo annotation')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default="../../models/yolov5l_remap.pth", help='weights checkpoint of model')
    parser.add_argument('--mapper', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--cocodir', type=str, default='../../MSCOCO2017')
    parser.add_argument('--smallscale', type=bool, default=False)
    parser.add_argument('--incremental_videos', type=bool, default=False)
    parser.add_argument('--train_whole', type=bool, default=False)
    parser.add_argument('--interm', action="store_true", help="whether model is splitted with intermediate layers")
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
    if args.opt == 'tp':
        inference_throughput(args)
    if args.opt == 'cluster':
        if args.random:
            random_cluster(args)
        else:
            offline_cluster(args)
