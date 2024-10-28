#!python3

import os
import sys
import types
import time
import datetime
import json
import copy
import math
import random
import tqdm
import glob
import gzip
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

import skimage.io
import skvideo.io
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict
import networkx

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.engine import launch as detectron2_launcher
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import BoxMode


thing_classes_coco = [['person'], ['car', 'bus', 'truck']]
thing_classes = ['person', 'vehicle']
assert len(thing_classes_coco) == len(thing_classes)


# https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py
def convert_torchvision_to_d2(pth_file, pkl_file):
    import pickle as pkl
    obj = torch.load(os.path.join(os.path.dirname(__file__), '..', 'configs', pth_file), map_location='cpu')
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if 'layer' not in k:
            k = 'stem.' + k
        for t in [1, 2, 3, 4]:
            k = k.replace('layer{}'.format(t), 'res{}'.format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace('bn{}'.format(t), 'conv{}.norm'.format(t))
        k = k.replace('downsample.0', 'shortcut')
        k = k.replace('downsample.1', 'shortcut.norm')
        print(old_k, '->', k)
        newmodel[k] = obj.pop(old_k).detach().numpy()
    with open(os.path.join(os.path.dirname(__file__), '..', 'configs', pkl_file), 'wb') as f:
        pkl.dump({'model': newmodel, '__author__': 'torchvision', 'matching_heuristics': True}, f)
    if obj:
        print('Unconverted keys:', obj.keys())


def get_coco_dicts_raw(args, split, segment=False):
    if split == 'valid':
        annotations_json = os.path.join(args.cocodir, 'annotations', 'instances_val2017.json')
    elif split == 'train':
        annotations_json = os.path.join(args.cocodir, 'annotations', 'instances_train2017.json')
    else:
        return None

    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    category_id_remap, thing_classes_coco_raw = {}, []
    for i, cat in enumerate(annotations['categories']):
        category_id_remap[cat['id']] = i
        thing_classes_coco_raw.append(cat['name'])
    coco_dicts_raw = {}
    images_dir = os.path.join(args.cocodir, 'images', 'val2017' if split == 'valid' else 'train2017')
    for im in annotations['images']:
        coco_dicts_raw[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
    for ann in annotations['annotations']:
        assert ann['category_id'] in category_id_remap
        coco_dicts_raw[ann['image_id']]['annotations'].append({'bbox': ann['bbox'], 'iscrowd': ann['iscrowd'], 'bbox_mode': BoxMode.XYWH_ABS, 'segmentation': ann['segmentation'] if segment else [], 'area': ann['area'], 'category_id': category_id_remap[ann['category_id']]})
    coco_dicts_raw = list(coco_dicts_raw.values())
    coco_dicts_raw = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts_raw))
    if args.smallscale:
        random.shuffle(coco_dicts_raw)
        coco_dicts_raw = coco_dicts_raw[: len(coco_dicts_raw) // 100]
    for i in range(0, len(coco_dicts_raw)):
        coco_dicts_raw[i]['image_id'] = i + 1
    count_images, count_bboxes = len(coco_dicts_raw), sum(map(lambda ann: len(ann['annotations']), coco_dicts_raw))
    print('MSCOCO-2017 %s: %d images, %d bboxes' % (split, count_images, count_bboxes))
    return coco_dicts_raw, thing_classes_coco_raw


def get_coco_dicts(args, split, segment=False):
    if split == 'valid':
        annotations_json = os.path.join(args.cocodir, 'instances_val2017.json')
    elif split == 'train':
        annotations_json = os.path.join(args.cocodir, 'instances_train2017.json')
    else:
        return None

    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    category_id_remap = {}
    for cat in annotations['categories']:
        for i in range(0, len(thing_classes_coco)):
            if cat['name'] in thing_classes_coco[i]:
                category_id_remap[cat['id']] = i
    coco_dicts = {}
    images_dir = os.path.join(args.cocodir, 'images', 'val2017' if split == 'valid' else 'train2017')
    for im in annotations['images']:
        coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
    for ann in annotations['annotations']:
        if not ann['category_id'] in category_id_remap:
            continue
        coco_dicts[ann['image_id']]['annotations'].append({'bbox': ann['bbox'], 'iscrowd': ann['iscrowd'], 'bbox_mode': BoxMode.XYWH_ABS, 'segmentation': ann['segmentation'] if segment else [], 'area': ann['area'], 'category_id': category_id_remap[ann['category_id']]})
    coco_dicts = list(coco_dicts.values())
    coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
    if args.smallscale:
        random.shuffle(coco_dicts)
        coco_dicts = coco_dicts[: len(coco_dicts) // 100]
    for i in range(0, len(coco_dicts)):
        coco_dicts[i]['image_id'] = i + 1
    count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('MSCOCO-2017 %s: %d images, %d bboxes' % (split, count_images, count_bboxes))
    return coco_dicts


def simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start
    if self.split_batch <= 1:
        loss_dict = self.model(data)
    else:
        # break down each batch in case VRAM is limited
        assert 0 == (len(data) % self.split_batch)
        loss_dict_chunks = []
        for i in range(0, len(data) // self.split_batch):
            loss_dict_chunks.append(self.model(data[i * self.split_batch : (i + 1) * self.split_batch]))
        loss_dict = {k: torch.stack([loss_dict_chunk_i[k] for loss_dict_chunk_i in loss_dict_chunks]).mean() for k in loss_dict_chunks[0]}
    loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
    if isinstance(loss_dict, torch.Tensor):
        losses = loss_dict
        loss_dict = {'total_loss': loss_dict}
    else:
        losses = sum(loss_dict.values())
    self.optimizer.zero_grad()
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    self.optimizer.step()
    if not hasattr(self, 'loss_history'):
        self.loss_history = []
    if not hasattr(self, 'lr_history'):
        self.lr_history = []
    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


class FinetuneTrainer(DefaultTrainer):
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
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)


def train_eval_remap(args):
    cfg_str = {
        'r50-c4-3x': 'COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
        'r50-fpn-3x': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'r101-fpn-3x': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
        'x101-fpn-3x': 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    }
    assert args.model in ['r50-c4-3x', 'r50-fpn-3x', 'r101-fpn-3x', 'x101-fpn-3x', 'r18-fpn-3x', 'r34-fpn-3x', 'r152-fpn-3x', 'x50-fpn-3x', 'x101-64x4d-fpn-3x']
    if args.model in ['r50-fpn-3x', 'r50-c4-3x', 'r101-fpn-3x', 'x101-fpn-3x']:
        args.det_cfg = model_zoo.get_config_file(cfg_str[args.model])
    # models not in model zoo
    elif args.model == 'r18-fpn-3x':
        args.det_cfg = os.path.join(os.path.dirname(__file__), '..', 'configs', 'faster_rcnn_R_18_FPN_3x.yaml')
    elif args.model == 'r34-fpn-3x':
        args.det_cfg = os.path.join(os.path.dirname(__file__), '..', 'configs', 'faster_rcnn_R_34_FPN_3x.yaml')
    elif args.model == 'r152-fpn-3x':
        args.det_cfg = os.path.join(os.path.dirname(__file__), '..', 'configs', 'faster_rcnn_R_152_FPN_3x.yaml')
    elif args.model == 'x50-fpn-3x':
        args.det_cfg = os.path.join(os.path.dirname(__file__), '..', 'configs', 'faster_rcnn_X_50_32x4d_FPN_3x.yaml')
    else:
        print('model not specified:', args.model)
        exit(1)
    output_dir = os.path.join(os.path.dirname(__file__), 'finetune_output_base_' + args.model.replace('-', '_'))

    DatasetCatalog.register('mscoco2017_train_remap', lambda: get_coco_dicts(args, 'train'))
    DatasetCatalog.register('mscoco2017_valid_remap', lambda: get_coco_dicts(args, 'valid'))
    MetadataCatalog.get('mscoco2017_train_remap').thing_classes = thing_classes
    MetadataCatalog.get('mscoco2017_valid_remap').thing_classes = thing_classes

    cfg = get_cfg()
    cfg.merge_from_file(args.det_cfg)
    if args.model in ['r50-fpn-3x', 'r50-c4-3x', 'r101-fpn-3x', 'x101-fpn-3x']:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_str[args.model])
    if not args.ckpt is None:
        cfg.MODEL.WEIGHTS = args.ckpt
    assert os.access(cfg.MODEL.WEIGHTS, os.R_OK), 'checkpoint not readable: ' + str(cfg.MODEL.WEIGHTS)
    cfg.OUTPUT_DIR = output_dir

    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.3
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = ('mscoco2017_train_remap',)
    cfg.DATASETS.TEST = ('mscoco2017_valid_remap',)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print('- image batch size:', cfg.SOLVER.IMS_PER_BATCH)
    print('- roi batch size:', cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
    print('- base lr:', cfg.SOLVER.BASE_LR)
    print('- lr warmpup iteration:', cfg.SOLVER.WARMUP_ITERS)
    print('- lr schedule gamma:', cfg.SOLVER.GAMMA)
    print('- lr schedule steps:', cfg.SOLVER.STEPS)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 180
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 250
    trainer = FinetuneTrainer(cfg)

    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.split_batch = args.split_batch
    trainer._trainer.run_step = types.MethodType(simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=args.resume)
    print('trainer initialized')

    prefix = 'mscoco2017_remap_%s' % args.model
    results_0 = OrderedDict()
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
    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(list(aps.keys()))
    mAPs = [aps[i]['bbox']['AP'] for i in iter_list]
    AP50s = [aps[i]['bbox']['AP50'] for i in iter_list]
    mAPs = np.array([x if not math.isnan(x) else 0.0 for x in mAPs], dtype=np.float32)
    AP50s = np.array([x if not math.isnan(x) else 0.0 for x in AP50s], dtype=np.float32)

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, AP50s / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, mAPs / 100, linestyle='-', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'AP50', 'mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlim(-100, max(iter_list) + 100)
    plt.ylim(-0.02, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP on MSCOCO-2017 Validation Split')

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
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))


def train_eval_coco_raw(args):
    model_list = {
        'r18-fpn-3x': 'faster_rcnn_R_18_FPN_3x.yaml',
        'r34-fpn-3x': 'faster_rcnn_R_34_FPN_3x.yaml',
        'r152-fpn-3x': 'faster_rcnn_R_152_FPN_3x.yaml',
        'x50-fpn-3x': 'faster_rcnn_X_50_32x4d_FPN_3x.yaml'
    }
    assert args.model in model_list, 'model not specified: ' + args.model
    args.det_cfg = os.path.join(os.path.dirname(__file__), '..', 'configs', model_list[args.model])
    output_dir = os.path.join(os.path.dirname(__file__), 'trainbase_output_' + args.model.replace('-', '_'))

    coco_dicts_train_raw, _ = get_coco_dicts_raw(args, 'train')
    coco_dicts_valid_raw, thing_classes_coco_raw = get_coco_dicts_raw(args, 'valid')
    DatasetCatalog.register('mscoco2017_train', lambda: coco_dicts_train_raw)
    DatasetCatalog.register('mscoco2017_valid', lambda: coco_dicts_valid_raw)
    MetadataCatalog.get('mscoco2017_train').thing_classes = thing_classes_coco_raw
    MetadataCatalog.get('mscoco2017_valid').thing_classes = thing_classes_coco_raw

    cfg = get_cfg()
    cfg.merge_from_file(args.det_cfg)
    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'configs', cfg.MODEL.WEIGHTS)
    cfg.OUTPUT_DIR = output_dir
    # roughly follow 3x schedule as of R-50 & R-101 in model zoo:
    # 118,000 images, ~36 epochs, batch=16, lr=0.02  -> 270,000 iters
    cfg.SOLVER.WARMUP_ITERS = min(args.iters // 2, cfg.SOLVER.WARMUP_ITERS)
    cfg.SOLVER.STEPS = (int(args.iters * 0.7), int(args.iters * 0.85))
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes_coco_raw)
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = ('mscoco2017_train',)
    cfg.DATASETS.TEST = ('mscoco2017_valid',)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print('- image batch size:', cfg.SOLVER.IMS_PER_BATCH)
    print('- roi batch size:', cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
    print('- base lr:', cfg.SOLVER.BASE_LR)
    print('- lr warmpup iteration:', cfg.SOLVER.WARMUP_ITERS)
    print('- lr schedule gamma:', cfg.SOLVER.GAMMA)
    print('- lr schedule steps:', cfg.SOLVER.STEPS)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 180
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 250
    trainer = FinetuneTrainer(cfg)

    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.split_batch = args.split_batch
    trainer._trainer.run_step = types.MethodType(simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=args.resume)
    print('trainer initialized')

    prefix = 'mscoco2017_%s' % args.model
    results_0 = OrderedDict()
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
    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(list(aps.keys()))
    mAPs = [aps[i]['bbox']['AP'] for i in iter_list]
    AP50s = [aps[i]['bbox']['AP50'] for i in iter_list]
    mAPs = np.array([x if not math.isnan(x) else 0.0 for x in mAPs], dtype=np.float32)
    AP50s = np.array([x if not math.isnan(x) else 0.0 for x in AP50s], dtype=np.float32)

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, AP50s / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, mAPs / 100, linestyle='-', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'AP50', 'mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlim(-100, max(iter_list) + 100)
    plt.ylim(-0.02, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP on MSCOCO-2017 (80 classes) Validation Split')

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
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))


if __name__ == '__main__':
    # https://download.pytorch.org/models/*.pth
    # convert_torchvision_to_d2('resnet18-f37072fd.pth', 'R-18.pkl'); exit(0)
    # convert_torchvision_to_d2('resnet34-b627a593.pth', 'R-34.pkl'); exit(0)
    # convert_torchvision_to_d2('resnet152-394f9c45.pth', 'R-152.pkl'); exit(0)
    # convert_torchvision_to_d2('resnext50_32x4d-7cdf4587.pth', 'X-50-32x4d.pkl'); exit(0)
    parser = argparse.ArgumentParser(description='Finetune Base Model on MSCOCO-2017 with Refined Classes')
    parser.add_argument('--opt', type=str, choices=['train', 'finetune'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--iters', type=int)
    parser.add_argument('--eval_interval', type=int)
    parser.add_argument('--cocodir', default='MSCOCO2017', type=str)
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--smallscale', default=False, type=bool)

    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--split_batch', type=int, default=1)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    print(args)

    if args.opt == 'finetune':
        if args.ddp_num_gpus <= 1:
            train_eval_remap(args)
        else:
            from detectron2.engine import launch
            launch(train_eval_remap, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    if args.opt == 'train':
        if args.ddp_num_gpus <= 1:
            train_eval_coco_raw(args)
        else:
            from detectron2.engine import launch
            launch(train_eval_coco_raw, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))


'''
conda deactivate && conda activate detectron2

nohup python base_detector_train.py --opt train --model r34-fpn-3x --iters 270000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 16 --ddp_num_gpus 4

python base_detector_train.py --opt finetune --model r34-fpn-3x --ckpt mscoco2017_r34-fpn-3x.pth --iters 40000 --eval_interval 2501 --cocodir ../../MSCOCO2017 --num_workers 4

python base_detector_train.py --opt finetune --model r34-fpn-3x --ckpt ../models/mscoco2017_r34-fpn-3x.pth --iters 120000 --eval_interval 6001 --cocodir ../../MSCOCO2017 --num_workers 4 --roi_batch_size 256 --lr 0.003

python base_detector_train.py --opt finetune --model r34-fpn-3x --ckpt larger_lr/mscoco2017_remap_r34-fpn-3x.pth --iters 15000 --eval_interval 2001 --roi_batch_size 256 --lr 0.0003 --cocodir ../../MSCOCO2017 --num_workers 4
'''
