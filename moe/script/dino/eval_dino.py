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
import contextlib

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata
import torchvision
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances, Boxes
from detectron2.modeling import build_model
from dino import *
from inference_server_simulate_dino import *
import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_detector_train import get_coco_dicts
from evaluation import evaluate_masked, evaluate_cocovalid
from dino import remove_splitted_boxes, split_into_four
import warnings
warnings.filterwarnings("ignore")

# Configuration constants
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


def convert_base_to_moe(state_dict, cfg):
    moe_state_dict = {}
    for key in state_dict.keys():
        if '.bbox_embed.' in key or '.class_embed.' in key:
            embed_position = key.find("embed.")
            # Find the position of the first dot (.) after "class_embed"
            dot_position = key.find(".", embed_position + len("embed."))
            for branch_id in range(cfg.MODEL.ADAPTIVE_BUDGET):
                new_key =  key[:dot_position] + f".experts.{branch_id}" + key[dot_position:]
                moe_state_dict[new_key] = deepcopy(state_dict[key])
        elif "enc_out_bbox_embed" in key or "enc_out_class_embed" in key:
            embed_position = key.find("embed.")
            # Find the position of the first dot (.) after "class_embed"
            dot_position = key.find(".", embed_position + len("embed"))
            for branch_id in range(cfg.MODEL.ADAPTIVE_BUDGET):
                new_key =  key[:dot_position] + f".experts.{branch_id}" + key[dot_position:]
                moe_state_dict[new_key] = deepcopy(state_dict[key])
        elif 'model.backbone.0.body.conv1' in key or 'model.backbone.0.body.bn1' in key or 'model.backbone.0.body.layer1' in key:
            body_position = key.find("body")
            dot_position = key.find(".", body_position + len("body."))
            for branch_id in range(cfg.MODEL.ADAPTIVE_BUDGET):
                new_key =  key[:dot_position] + f".experts.{branch_id}" + key[dot_position:]
                moe_state_dict[new_key] = deepcopy(state_dict[key])              
        else:
            moe_state_dict[key] = deepcopy(state_dict[key])
    state_dict = moe_state_dict
    return state_dict


class SemiRandomClient(torchdata.Dataset):
    def __init__(self, cfg, scale=1):
        super().__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.scale = scale

        self.images = []
        for video_id in video_id_list:
            inputdir = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'annotation', video_id
            )
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                _dicts = json.load(fp)
            for im in _dicts:
                im['md5'] = f'{video_id}_{im["file_name"]}'
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
        for i in tqdm.tqdm(range(len(self.images)), ascii=True, desc='Preloading images'):
            self.preloaded_images.append(self.read(i))

    def __len__(self):
        return len(self.images)

    def read(self, i):
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        image = torchvision.transforms.functional.resize(
            image, size=[int(self.scale * image.shape[1]), int(self.scale * image.shape[2])]
        )
        return {'image': image, 'height': height, 'width': width, 'video_id': self.images[i]['video_id']}

    def __getitem__(self, i):
        if self.preloaded_images is None:
            return self.read(i), self.images[i]
        else:
            return self.preloaded_images[i], self.images[i]

    @staticmethod
    def collate(batch):
        return batch


class COCOEvaluationDataset(torchdata.Dataset):
    def __init__(self, images, cfg):
        super().__init__()
        self.images = images
        self.input_format = cfg.INPUT.FORMAT
        self.aug = detectron2.data.transforms.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        image = torchvision.transforms.functional.resize(image, size=[image.shape[1], image.shape[2]])
        return {'image': image, 'height': height, 'width': width, 'video_id': 'coco'}, self.images[i]

    @staticmethod
    def collate(batch):
        return batch


def evaluate_coco(args):
    cfg = get_cfg()
    add_dino_config(cfg)
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), 'Checkpoint not readable'
    model = build_model(cfg)
    model = DINOServer.create_from_sup(model, args.budget, args.interm)

    try:
        state_dict = torch.load(args.ckpt)['model']
    except KeyError:
        state_dict = torch.load(args.ckpt)
    
    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    try:    
        model.load_state_dict(convert_base_to_moe(state_dict, cfg), strict=True)
    except:
        model.load_state_dict(state_dict, strict=True)

    if args.mapper is not None:
        mapper = torch.load(args.mapper)
        model.budget = mapper['budget']
        model.video_id_to_index = mapper['video_id_to_index']
        model.used_indices = mapper['used_indices']
        model.un_used_indices = mapper['un_used_indices']
    model.eval()

    images = get_coco_dicts(args, 'train')

    for i in range(len(images)):
        for j, ann in enumerate(images[i]['annotations']):
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x, y, w, h = ann['bbox']
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                images[i]['annotations'][j]['bbox'] = [x1, y1, x2, y2]
                images[i]['annotations'][j]['bbox_mode'] = BoxMode.XYXY_ABS

    loader = torchdata.DataLoader(
        COCOEvaluationDataset(images, cfg),
        batch_size=None, collate_fn=COCOEvaluationDataset.collate, shuffle=False, num_workers=2
    )

    detections = []

    for inputs, im in tqdm.tqdm(loader, total=len(images), ascii=True):
        im = deepcopy(im)
        im['annotations'] = []
        with torch.no_grad():
            instances = model([inputs])[0]['instances'].to('cpu')
            bbox = instances.pred_boxes.tensor
            score = instances.scores
            label = instances.pred_classes
            for i in range(len(label)):
                im['annotations'].append({
                    'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])
                })
        detections.append(im)

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results_ap = evaluate_cocovalid(args.cocodir, detections)

    print(results_ap['results']['weighted'])
    return results_ap['results']['weighted']


def evaluate_scenes100(args):
    cfg = get_cfg()
    add_dino_config(cfg)
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), 'Checkpoint not readable'
    
    model = build_model(cfg)
    model = DINOServer.create_from_sup(model, args.budget, args.interm)

    try:
        state_dict = torch.load(args.ckpt)['model']
    except KeyError:
        state_dict = torch.load(args.ckpt)

    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    try:    
        model.load_state_dict(convert_base_to_moe(state_dict, cfg), strict=True)
    except:
        model.load_state_dict(state_dict, strict=True)

    if args.mapper is not None:
        mapper = torch.load(args.mapper)
        model.budget = mapper['budget']
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

    torch.cuda.empty_cache()

    detections = {v: [] for v in video_id_list}
    t_total = time.time()

    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='Detecting'):
        det = deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            instances = model([inputs])[0]['instances'].to('cpu')
            det['instances'] = {
                'bbox': instances.pred_boxes.tensor,
                'score': instances.scores,
                'label': instances.pred_classes
            }
        detections[im['video_id']].append(det)

    t_total = time.time() - t_total
    print(f'{len(dataset)} finished in {t_total:.1f} seconds, throughput {len(dataset) / t_total:.3f} images/sec')

    results = {}
    for video_id in tqdm.tqdm(detections, ascii=True, desc='Evaluating'):
        for det in detections[video_id]:
            bbox, score, label = (
                det['instances']['bbox'].numpy().tolist(),
                det['instances']['score'].numpy().tolist(),
                det['instances']['label'].numpy().tolist()
            )
            for i in range(len(label)):
                det['annotations'].append({
                    'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [], 'category_id': label[i], 'score': score[i]
                })
            del det['instances']
            det['file_name'] = os.path.basename(det['file_name'])

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)

        results[video_id]['detections'] = detections

    categories = ['person', 'vehicle', 'overall', 'weighted']
    avg_result = {c: [] for c in categories}
    
    for video_id in results:
        ap = results[video_id]['results']
        for cat in categories:
            avg_result[cat].append([ap[cat][0], ap[cat][1]])

    for cat in categories:
        avg_result[cat] = np.array(avg_result[cat]) * 100.0
        print(f'{cat}: mAP {avg_result[cat][:, 0].mean():.4f}, AP50 {avg_result[cat][:, 1].mean():.4f}')

    return avg_result['weighted'][:, 0].mean(), avg_result['weighted'][:, 1].mean()


def evaluate_scenes100_split(args):
    cfg = get_cfg()
    add_dino_config(cfg)
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), 'Checkpoint not readable'

    model = build_model(cfg)

    try:
        state_dict = torch.load(args.ckpt)['model']
    except KeyError:
        state_dict = torch.load(args.ckpt)

    model.load_state_dict(state_dict, strict=True)

    if args.mapper is not None:
        mapper = torch.load(args.mapper)
        model.budget = mapper['budget']
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

    torch.cuda.empty_cache()

    detections = {v: [] for v in video_id_list}
    t_total = time.time()

    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='Detecting'):
        patch_inputs, offsets = split_into_four(inputs)
        det = deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            all_bboxes = []
            all_scores = []
            all_labels = []
            num_instances_from_patches = 0
            for i, patch_input in enumerate(patch_inputs):
                offset = offsets[i]
                instances = model([patch_input])[0]['instances'].to('cpu')
                if i < 4:
                    num_instances_from_patches += instances.pred_boxes.tensor.shape[0]
                    for j in range(instances.pred_boxes.tensor.shape[0]):
                        instances.pred_boxes.tensor[j][:2] /= 2
                        instances.pred_boxes.tensor[j][2:] /= 2
                        instances.pred_boxes.tensor[j][:2] += offset
                        instances.pred_boxes.tensor[j][2:] += offset

                all_bboxes.append(instances.pred_boxes.tensor)
                all_scores.append(instances.scores)
                all_labels.append(instances.pred_classes)

            all_bboxes = torch.cat(all_bboxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            keep = remove_splitted_boxes(all_bboxes, inputs['height'], inputs['width'], num_instances_from_patches)
            all_bboxes, all_scores, all_labels = all_bboxes[keep], all_scores[keep], all_labels[keep]

            keep = batched_nms(all_bboxes, all_scores, all_labels, 0.5)
            all_bboxes, all_scores, all_labels = all_bboxes[keep], all_scores[keep], all_labels[keep]

            _, keep = torch.topk(all_scores, 300)
            all_bboxes, all_scores, all_labels = all_bboxes[keep], all_scores[keep], all_labels[keep]

            boxes = Boxes(all_bboxes)
            all_instances = Instances(instances.image_size)
            all_instances.set('pred_boxes', boxes)
            all_instances.set('scores', all_scores)
            all_instances.set('pred_classes', all_labels)

            det['instances'] = {'bbox': all_bboxes, 'score': all_scores, 'label': all_labels}

        detections[im['video_id']].append(det)

    t_total = time.time() - t_total
    print(f'{len(dataset)} finished in {t_total:.1f} seconds, throughput {len(dataset) / t_total:.3f} images/sec')

    results = {}
    for video_id in tqdm.tqdm(detections, ascii=True, desc='Evaluating'):
        for det in detections[video_id]:
            bbox, score, label = (
                det['instances']['bbox'].numpy().tolist(),
                det['instances']['score'].numpy().tolist(),
                det['instances']['label'].numpy().tolist()
            )
            for i in range(len(label)):
                det['annotations'].append({
                    'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [], 'category_id': label[i], 'score': score[i]
                })
            del det['instances']
            det['file_name'] = os.path.basename(det['file_name'])

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)

        results[video_id]['detections'] = detections

    categories = ['person', 'vehicle', 'overall', 'weighted']
    avg_result = {c: [] for c in categories}

    for video_id in results:
        ap = results[video_id]['results']
        for cat in categories:
            avg_result[cat].append([ap[cat][0], ap[cat][1]])

    for cat in categories:
        avg_result[cat] = np.array(avg_result[cat]) * 100.0
        print(f'{cat}: mAP {avg_result[cat][:, 0].mean():.4f}, AP50 {avg_result[cat][:, 1].mean():.4f}')

    return avg_result['weighted'][:, 0].mean(), avg_result['weighted'][:, 1].mean()


def inference_throughput(args):
    cfg = get_cfg()
    add_dino_config(cfg)
    
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), f'Checkpoint not readable: {args.ckpt}'
    model = build_model(cfg)
    model = DINOServer.create_from_sup(model, args.budget, args.interm)
    
    try:
        state_dict = torch.load(args.ckpt)['model']
    except KeyError:
        state_dict = torch.load(args.ckpt)
    
    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    try:    
        model.load_state_dict(convert_base_to_moe(state_dict, cfg), strict=True)
    except:
        model.load_state_dict(state_dict, strict=True)
    
    if args.mapper is not None:
        mapper = torch.load(args.mapper)
        model.budget = mapper['budget']
        model.video_id_to_index = mapper['video_id_to_index']
        model.used_indices = mapper['used_indices']
        model.un_used_indices = mapper['un_used_indices']
        
    model.eval()
    dataset = SemiRandomClient(cfg)
    dataset.images = [img for img in dataset.images if img['video_id'] == args.id]
    dataset.images = sorted(dataset.images, key=lambda x: x['file_name'])[:10]
    dataset.preload()

    gc.collect()
    torch.cuda.empty_cache()

    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(N2 + N1), ascii=True):
            if i == N1:
                t_start = time.time()
            if i == N2:
                t_total = time.time() - t_start
            model([dataset[i % len(dataset)][0]])
    
    throughput = (N2 - N1) / t_total
    print(f'{throughput:.3f} images/s, {1000 / throughput:.3f} ms/image')


def inference_throughput_batch(args):
    cfg = get_cfg()
    add_dino_config(cfg)

    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), f'Checkpoint not readable: {args.ckpt}'
    model = build_model(cfg)
    model = DINOServer.create_from_sup(model, args.budget, args.interm)

    try:
        state_dict = torch.load(args.ckpt)['model']
    except KeyError:
        state_dict = torch.load(args.ckpt)

    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    try:    
        model.load_state_dict(convert_base_to_moe(state_dict, cfg), strict=True)
    except:
        model.load_state_dict(state_dict, strict=True)

    if args.mapper is not None:
        mapper = torch.load(args.mapper)
        model.budget = mapper['budget']
        model.video_id_to_index = mapper['video_id_to_index']
        model.used_indices = mapper['used_indices']
        model.un_used_indices = mapper['un_used_indices']

    model.eval()
    dataset = SemiRandomClient(cfg)
    dataset.images = [img for img in dataset.images if img['video_id'] == args.id]
    dataset.images = sorted(dataset.images, key=lambda x: x['file_name'])[:10]
    dataset.preload()

    images = [img[0] for img in dataset]
    for i, img in enumerate(images):
        im_batch = []
        for video_id in video_id_list[:args.image_batch_size]:
            im_copy = deepcopy(img)
            im_copy['video_id'] = video_id  # Assign different video_id
            im_batch.append(im_copy)
        images[i] = im_batch

    del dataset

    gc.collect()
    torch.cuda.empty_cache()

    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(N2 + N1), ascii=True):
            if i == N1:
                t_start = time.time()
            if i == N2:
                t_total = time.time() - t_start
            model(images[i % len(images)])

    throughput = (N2 - N1) / t_total
    print(f'{throughput:.3f} images/s, {1000 / throughput:.3f} ms/image')


def evaluate_individual_scenes100(args):
    global video_id_list
    print(video_id_list)

    video_id_list_copy = deepcopy(video_id_list)
    weighted_APs = {idx: [] for idx in video_id_list}

    for video_id in video_id_list_copy:
        video_id_list = [video_id]
        args.ckpt = os.path.join(
            os.path.dirname(__file__),
            "finetune_bs2_lr0.0001_teacherx2split_conf0.3_continue8k_equal_iters_20",
            video_id,
            "adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.20.pth"
        )
        weighted_APs[video_id] = evaluate_scenes100(args)
        print(f"Video {video_id}: mAP {weighted_APs[video_id][0]:.4f}, AP50 {weighted_APs[video_id][1]:.4f}")

    mAPs = np.array([weighted_APs[idx][0] for idx in video_id_list_copy])
    AP50s = np.array([weighted_APs[idx][1] for idx in video_id_list_copy])

    print(f"Avg mAP: {np.mean(mAPs):.4f}, Avg AP50: {np.mean(AP50s):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Script')

    parser.add_argument('--opt', type=str, default='server', help='option')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default="../../models/yolov5s_remap.pth", help='model weights checkpoint')
    parser.add_argument('--mapper', type=str, default=None, help='mapper checkpoint')
    parser.add_argument('--config', type=str, default="../../configs/yolov5s.yaml", help='model config file')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--cocodir', type=str, default='../../../mscoco')
    parser.add_argument('--interm', action="store_true", help="whether model is splitted with intermediate layers")

    parser.add_argument('--preload', type=bool, default=False)
    parser.add_argument('--smallscale', type=bool, default=False)
    parser.add_argument('--instances', type=int, default=1)
    parser.add_argument('--val_set', type=str, choices=['scenes100', 'coco'], default='scenes100')

    args = parser.parse_args()
    print(args)
    
    if args.id != '':
        video_id_list = [args.id]

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
python eval_dino.py --opt tp --ckpt ../../models/dino_5scale_remap_orig.pth --scale 1 --budget 10 --id 001 --image_batch_size 4
'''
