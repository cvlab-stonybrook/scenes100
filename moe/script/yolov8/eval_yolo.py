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
import contextlib
import logging
import weakref
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import torchvision
from torchvision.ops.boxes import batched_nms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances, Boxes
from detectron2.config import get_cfg

from yolov8 import *
from inference_server_simulate_yolov8 import YOLOServer

# Adding parent directory to sys.path for importing custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from evaluation import evaluate_masked, evaluate_cocovalid

# Configuration and constants
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


class SemiRandomClient(torchdata.Dataset):
    """Dataset class for semi-randomly sampled images."""

    def __init__(self, cfg, scale=1):
        super(SemiRandomClient, self).__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.scale = scale
        assert self.input_format == 'BGR'

        self.images = []
        for video_id in video_id_list:
            inputdir = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'scenes100', 'annotation', video_id
            )
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                _dicts = json.load(fp)
            for im in _dicts:
                # Add md5 hash for pseudo-random shuffling
                im['md5'] = f'{video_id}_{im["file_name"]}'
                im['md5'] = hashlib.md5(im['md5'].encode('utf-8')).hexdigest()
                im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))
                im['video_id'] = video_id
            self.images.extend(_dicts)
        self.images.sort(key=lambda x: x['md5'])
        self.preloaded_images = None

    def preload(self):
        """Preload all images into memory."""
        if self.preloaded_images is not None:
            return
        self.preloaded_images = []
        for i in tqdm.tqdm(range(0, len(self.images)), ascii=True, desc='preloading images'):
            self.preloaded_images.append(self.read(i))

    def __len__(self):
        return len(self.images)

    def read(self, i):
        """Read and preprocess an image."""
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        image = torchvision.transforms.functional.resize(image, size=[int(self.scale * image.shape[1]), int(self.scale * image.shape[2])])
        return {'image': image, 'height': height, 'width': width, 'video_id': self.images[i]['video_id']}

    def __getitem__(self, i):
        """Get a preloaded image or read it if not preloaded."""
        if self.preloaded_images is None:
            return self.read(i), self.images[i]
        else:
            return self.preloaded_images[i], self.images[i]

    @staticmethod
    def collate(batch):
        """Custom collation function for DataLoader."""
        return batch


class COCOEvaluationDataset(torch.utils.data.Dataset):
    """Dataset class for COCO evaluation."""

    def __init__(self, images, cfg):
        super(COCOEvaluationDataset, self).__init__()
        self.images = images
        self.input_format = cfg.INPUT.FORMAT
        self.aug = detectron2.data.transforms.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        """Read and preprocess an image for evaluation."""
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        image = torchvision.transforms.functional.resize(image, size=[image.shape[1], image.shape[2]])
        return {'image': image, 'height': height, 'width': width, 'video_id': 'coco'}, self.images[i]

    @staticmethod
    def collate(batch):
        """Custom collation function for DataLoader."""
        return batch


def evaluate_coco(args):
    """Evaluate model on COCO dataset."""
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), 'checkpoint not readable: ' + args.ckpt

    try:
        model = load_yolov8(args.config, args.ckpt)
    except:
        model = load_yolov8(args.config)
        model = YOLOServer.create_from_sup(model, args.budget, args.split_list)
        model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
        if args.budget > 1:
            mapper = torch.load(args.mapper)
            model.video_id_to_index = mapper['video_id_to_index']
            model.used_indices = mapper['used_indices']
            model.un_used_indices = mapper['un_used_indices']

    model.eval()

    images = get_coco_dicts(args, 'valid')

    # Convert bounding boxes to XYXY format
    for i in range(len(images)):
        for j, ann in enumerate(images[i]['annotations']):
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x, y, w, h = ann['bbox']
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                images[i]['annotations'][j]['bbox'] = [x1, y1, x2, y2]
                images[i]['annotations'][j]['bbox_mode'] = BoxMode.XYXY_ABS

    loader = torch.utils.data.DataLoader(
        COCOEvaluationDataset(images, cfg),
        batch_size=None, collate_fn=COCOEvaluationDataset.collate, shuffle=False, num_workers=2
    )

    detections = []

    for input, im in tqdm.tqdm(loader, total=len(images), ascii=True):
        im = deepcopy(im)

        im['annotations'] = []
        with torch.no_grad():
            instances = model([input])[0]['instances'].to('cpu')
            # Extract and store detections
            bbox = instances.pred_boxes.tensor
            score = instances.scores
            label = instances.pred_classes
            for i in range(0, len(label)):
                im['annotations'].append({
                    'bbox': list(map(float, bbox[i])),
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [],
                    'category_id': int(label[i]),
                    'score': float(score[i])
                })
        detections.append(im)

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results_AP = evaluate_cocovalid(args.cocodir, detections)

    print(results_AP['results']['weighted'])
    return results_AP['results']['weighted']


def evaluate_scenes100(args):
    """Evaluate model on scenes100 dataset."""
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))

    try:
        model = load_yolov8(args.config, args.ckpt)
    except:
        model = load_yolov8(args.config)
        model = YOLOServer.create_from_sup(model, args.budget, args.split_list)
        model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
        if args.budget > 1:
            mapper = torch.load(args.mapper)
            model.video_id_to_index = mapper['video_id_to_index']
            model.used_indices = mapper['used_indices']
            model.un_used_indices = mapper['un_used_indices']

    model.eval()
    dataset = SemiRandomClient(cfg, args.scale)
    if args.preload:
        dataset.preload()
    loader = torchdata.DataLoader(
        dataset, batch_size=None, collate_fn=SemiRandomClient.collate, shuffle=False, num_workers=1
    )
    gc.collect()
    torch.cuda.empty_cache()

    detections = {v: [] for v in video_id_list}
    t_total = time.time()

    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='detecting'):
        det = deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            instances = model.inference([inputs])[0]['instances'].to('cpu')
            det['instances'] = {
                'bbox': instances.pred_boxes.tensor,
                'score': instances.scores,
                'label': instances.pred_classes
            }
            ratio = inputs['image'].shape[1] / inputs['height']

        detections[im['video_id']].append(det)

    t_total = time.time() - t_total
    print('%d finished in %.1f seconds, throughput %.3f images/sec' % (len(dataset), t_total, len(dataset) / t_total))

    results = {}
    for video_id in tqdm.tqdm(detections, ascii=True, desc='evaluating'):
        for det in detections[video_id]:
            bbox, score, label = det['instances']['bbox'].numpy().tolist(), det['instances']['score'].numpy().tolist(), det['instances']['label'].numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({
                    'bbox': bbox[i],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [],
                    'category_id': label[i],
                    'score': score[i]
                })
            del det['instances']
            det['file_name'] = os.path.basename(det['file_name'])
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)

    categories = ['person', 'vehicle', 'overall', 'weighted']
    avg_result = {c: [] for c in categories}
    for video_id in results:
        AP = results[video_id]['results']
        for cat in categories:
            avg_result[cat].append([AP[cat][0], AP[cat][1]])
    for cat in categories:
        avg_result[cat] = np.array(avg_result[cat]) * 100.0
        print('%s: mAP %.4f, AP50 %.4f' % (cat, avg_result[cat][:, 0].mean(), avg_result[cat][:, 1].mean()))
    return [avg_result['weighted'][:, 0].mean(), avg_result['weighted'][:, 1].mean()]


def evaluate_scenes100_split(args):
    """Evaluate model on scenes100 dataset with splitted images."""
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))

    try:
        model = load_yolov8(args.config, args.ckpt)
    except:
        model = load_yolov8(args.config)
        model = YOLOServer.create_from_sup(model, args.budget, args.split_list)
        model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
        if args.budget > 1:
            mapper = torch.load(args.mapper)
            model.video_id_to_index = mapper['video_id_to_index']
            model.used_indices = mapper['used_indices']
            model.un_used_indices = mapper['un_used_indices']

    model.eval()
    dataset = SemiRandomClient(cfg, args.scale)
    if args.preload:
        dataset.preload()
    loader = torchdata.DataLoader(
        dataset, batch_size=None, collate_fn=SemiRandomClient.collate, shuffle=False, num_workers=1
    )
    gc.collect()
    torch.cuda.empty_cache()

    detections = {v: [] for v in video_id_list}
    t_total = time.time()

    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='detecting'):
        patch_inputs, offsets = split_into_four(inputs)
        ratio = inputs['image'].shape[1] / inputs['height']

        det = deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            all_bboxes = []
            all_scores = []
            all_labels = []
            num_instances_from_patches = 0
            for i, patch_input in enumerate(patch_inputs):
                offset = offsets[i]
                instances = model.inference([patch_input])[0]['instances'].to('cpu')
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

            all_bboxes = torch.cat(all_bboxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Remove boxes on the border
            keep = remove_splitted_boxes(all_bboxes, inputs['height'], inputs['width'], num_instances_from_patches)
            all_bboxes = all_bboxes[keep]
            all_scores = all_scores[keep]
            all_labels = all_labels[keep]

            # Apply Non-Maximum Suppression (NMS)
            keep = batched_nms(all_bboxes, all_scores, all_labels, 0.45)
            all_bboxes = all_bboxes[keep]
            all_scores = all_scores[keep]
            all_labels = all_labels[keep]

            boxes = Boxes(all_bboxes)
            all_instances_dict = {'pred_boxes': boxes, 'scores': all_scores, 'pred_classes': all_labels}
            all_instances = Instances(instances.image_size)
            for (k, v) in all_instances_dict.items():
                all_instances.set(k, v)

            DetectionModel.draw(
                inputs['image'][(2, 1, 0), :, :] / 255, all_instances, 'frcnn', ratio=ratio, file_name='combine.png'
            )
            det['instances'] = {
                'bbox': all_bboxes,
                'score': all_scores,
                'label': all_labels
            }

        detections[im['video_id']].append(det)

    t_total = time.time() - t_total
    print('%d finished in %.1f seconds, throughput %.3f images/sec' % (len(dataset), t_total, len(dataset) / t_total))

    results = {}
    for video_id in tqdm.tqdm(detections, ascii=True, desc='evaluating'):
        for det in detections[video_id]:
            bbox, score, label = det['instances']['bbox'].numpy().tolist(), det['instances']['score'].numpy().tolist(), det['instances']['label'].numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({
                    'bbox': bbox[i],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [],
                    'category_id': label[i],
                    'score': score[i]
                })
            del det['instances']
            det['file_name'] = os.path.basename(det['file_name'])
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)

    categories = ['person', 'vehicle', 'overall', 'weighted']
    avg_result = {c: [] for c in categories}
    for video_id in results:
        AP = results[video_id]['results']
        for cat in categories:
            avg_result[cat].append([AP[cat][0], AP[cat][1]])
    for cat in categories:
        avg_result[cat] = np.array(avg_result[cat]) * 100.0
        print('%s: mAP %.4f, AP50 %.4f' % (cat, avg_result[cat][:, 0].mean(), avg_result[cat][:, 1].mean()))
    return [avg_result['weighted'][:, 0].mean(), avg_result['weighted'][:, 1].mean()]


def remove_splitted_boxes(all_bboxes, h, w, limit):
    """Remove bounding boxes that are split across patches."""
    keep = []
    for i in range(all_bboxes.shape[0]):
        if i >= limit:
            keep.append(i)
        elif (
            abs(all_bboxes[i][0] - w / 2) > 5 and abs(all_bboxes[i][1] - h / 2) > 5 and
            abs(all_bboxes[i][2] - w / 2) > 5 and abs(all_bboxes[i][3] - h / 2) > 5
        ):
            keep.append(i)
    return keep


def split_into_four(inputs):
    """Split input image into four patches and resize them."""
    H_orig, W_orig = inputs['height'], inputs['width']
    img_tensor = inputs['image']

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
        patch_inputs = {
            'image': patch,
            'height': inputs['height'],
            'width': inputs['width'],
            'video_id': inputs['video_id']
        }
        all_inputs.append(patch_inputs)
    all_inputs.append(inputs)
    offsets = [[0, 0], [W_orig // 2, 0], [0, H_orig // 2], [W_orig // 2, H_orig // 2], [0, 0]]
    return all_inputs, offsets


def inference_throughput(args):
    """Measure inference throughput for a single image."""
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    model = load_yolov8(args.config, args.ckpt)
    model = YOLOServer.create_from_sup(model, args.budget, args.split_list)
    if args.budget > 1 and args.mapper != '':
        mapper = torch.load(args.mapper)
        model.video_id_to_index = mapper['video_id_to_index']
    model.eval()
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


def inference_throughput_batch(args):
    """Measure inference throughput for batched images."""
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    model = load_yolov8(args.config, args.ckpt)
    model = YOLOServer.create_from_sup(model, args.budget, args.split_list)
    if args.budget > 1 and args.mapper != '':
        mapper = torch.load(args.mapper)
        model.video_id_to_index = mapper['video_id_to_index']
    model.eval()
    dataset = SemiRandomClient(cfg)
    dataset.images = list(filter(lambda x: x['video_id'] == args.id, dataset.images))
    dataset.images = sorted(dataset.images, key=lambda x: x['file_name'])[:10]
    dataset.preload()

    images = [im[0] for im in dataset]
    for i in range(0, len(images)):
        im_batch = []
        for v in video_id_list[:args.image_batch_size]:
            im_batch.append(deepcopy(images[i]))
            im_batch[-1]['video_id'] = v  # Use different video_id for each image
        images[i] = im_batch
    del dataset

    gc.collect()
    torch.cuda.empty_cache()
    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            model.inference(images[i % len(images)])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


def evaluate_individual_scenes100(args):
    """Evaluate model on individual scenes100 videos."""
    global video_id_list
    print(video_id_list)
    video_id_list_copy = deepcopy(video_id_list)
    weighted_APs = {idx: [] for idx in video_id_list}
    for idx in video_id_list_copy:
        video_id_list = [idx]
        args.ckpt = os.path.join(
            os.path.dirname(__file__),
            "finetune_bs28_lr0.0001_teacherx2_conf0.4_continue24k_equal_iters", idx,
            "adaptive_partial_server_yolov3_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.100.pth"
        )
        weighted_APs[idx] = evaluate_scenes100(args)
        print(f"Video {idx}: mAP {weighted_APs[idx][0]}, AP50 {weighted_APs[idx][1]}")
    mAPs = np.array([weighted_APs[idx][0] for idx in video_id_list_copy])
    AP50s = np.array([weighted_APs[idx][1] for idx in video_id_list_copy])
    print(f"Avg mAP: {np.mean(mAPs)}, Avg AP50: {np.mean(AP50s)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--opt', type=str, default='server', help='option')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default="../../models/yolov8s_remap.pth", help='weights checkpoint of model')
    parser.add_argument('--mapper', type=str, default="", help='mapper checkpoint of model')
    parser.add_argument('--config', type=str, default="../../configs/yolov8.yaml", help='config of model')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--cocodir', type=str, default='../../MSCOCO2017')

    parser.add_argument('--preload', type=bool, default=False)
    parser.add_argument('--smallscale', type=bool, default=False)
    parser.add_argument('--instances', type=int, default=1)
    parser.add_argument('--val_set', type=str, choices=['scenes100', 'coco'], default='scenes100')
    parser.add_argument('--split_list', type=int, nargs='+')

    args = parser.parse_args()
    print(args)

    if args.opt == 'server':
        assert args.instances > 0
        if args.instances == 1:
            if args.val_set == 'scenes100': 
                evaluate_scenes100(args)
            else:
                evaluate_coco(args)
    if args.opt == 'tp':
        inference_throughput_batch(args)

'''

python eval_yolo.py --opt tp --config ../../configs/yolov8s.yaml --ckpt ../../models/yolov8s_remap.pth --budget 100 --id 001 --split_list 0 1 2 3 4 -1 --image_batch_size 1

'''