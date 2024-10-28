#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
import copy
import gzip
import math
import random
import tqdm
import glob
import psutil
import argparse
import enum
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

import sklearn.utils
from sklearn.mixture import GaussianMixture

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


from dino import *
from util.get_param_dicts import get_param_dict
from cfg_to_args import cfg_to_args 

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output')


import torch.utils.data

class AnnotationType(enum.IntEnum):
    RELIABLE = 0
    PSEUDO   = 1

class RefineVisualizeDataset(torch.utils.data.Dataset):
    def __init__(self, video_id, refined_dict, anno_models):
        super(RefineVisualizeDataset, self).__init__()
        self.dst = TrainingFrames(video_id)
        self.refined_dict = refined_dict
        self.anno_models = ' + '.join(anno_models)
        self.font_label = None
        self.font_title = None
    def __len__(self):
        return len(self.dst)
    def __getitem__(self, i):
        if self.font_label is None:
            self.font_label = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=18)
            self.font_title = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=60)
        f = Image.fromarray(self.dst[i][0])
        draw = ImageDraw.Draw(f)
        for ann in self.refined_dict[i]['annotations']:
            x1, y1, x2, y2 = ann['bbox']
            c = bbox_rgbs[ann['category_id']]
            draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=c, width=5)
            draw.text((x1 + 2, y1 + 2), '%d %s' % (ann['category_id'], thing_classes[ann['category_id']]), fill=c, font=self.font_label)
        draw.text((2, 2), 'Pseudo: %s' % self.anno_models, fill='#000000', stroke_width=3, font=self.font_title)
        draw.text((2, 2), 'Pseudo: %s' % self.anno_models, fill='#FFFFFF', stroke_width=1, font=self.font_title)
        return np.array(f)


def _graph_refine(params):
    _dict_json, _args, desc = params['dict'], params['args'], params['desc']
    count_bboxes = 0
    for annotations in tqdm.tqdm(_dict_json, ascii=True, desc='refining chunk ' + desc):
        G = networkx.Graph()
        [G.add_node(i) for i in range(0, len(annotations['annotations']))]
        for i in range(0, len(annotations['annotations'])):
            for j in range(i, len(annotations['annotations'])):
                iou_value = IoU(annotations['annotations'][i]['bbox'], annotations['annotations'][j]['bbox'])
                if annotations['annotations'][i]['category_id'] == annotations['annotations'][j]['category_id'] and iou_value > _args.refine_iou_thres:
                    G.add_edge(i, j, iou=iou_value)
        subs = list(networkx.algorithms.components.connected_components(G))

        anns_refine = []
        for sub_nodes in subs:
            max_degree, max_degree_n = -1, -1
            for n in sub_nodes:
                D = sum(map(lambda t: t[2], list(G.edges(n, data='iou'))))
                if D > max_degree:
                    max_degree, max_degree_n = D, n
            anns_refine.append(annotations['annotations'][max_degree_n])
        annotations['annotations'] = anns_refine
        if 'det_count' in annotations: del annotations['det_count']
        if 'sot_count' in annotations: del annotations['sot_count']
        annotations['bbox_count'] = len(annotations['annotations'])
        count_bboxes += annotations['bbox_count']
    return _dict_json, count_bboxes

def refine_annotations(args, visualize=False):
    dst = TrainingFrames(args.id)
    imagedir = os.path.join(dst.lmdb_path, 'jpegs')
    labeldir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label'))
    det_filelist, sot_filelist = [], []
    for m in args.anno_models:
        det_filelist.append(os.path.normpath(os.path.join(labeldir, '%s_detect_%s.json.gz' % (args.id, m))))
        sot_filelist.append(os.path.normpath(os.path.join(labeldir, '%s_detect_%s_DiMP.json.gz' % (args.id, m))))
    for f in det_filelist + sot_filelist:
        assert os.access(f, os.R_OK), '%s not readable' % f
    if args.fn_max_samples > 0:
        fn_file = os.path.join(labeldir, '%s_false_negative_mining_objthres0.9900.json.gz' % args.id)
        assert os.access(fn_file, os.R_OK), 'false negative mining result file not readable'

    # collate bboxes from tracking & detection
    dict_json = []
    for i in range(0, len(dst)):
        dict_json.append({'file_name': os.path.normpath(os.path.join(imagedir, dst.ifilelist[i])), 'image_id': i, 'height': dst.meta['meta']['video']['H'], 'width': dst.meta['meta']['video']['W'], 'annotations': [], 'det_count': 0, 'sot_count': 0, 'fn_count': 0})

    for f in det_filelist:
        print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
        with gzip.open(f, 'rt') as fp:
            dets = json.loads(fp.read())['dets']
        assert len(dets) == len(dict_json), 'detection & dataset mismatch'
        for i in range(0, len(dets)):
            for j in range(0, len(dets[i]['score'])):
                if dets[i]['score'][j] < args.refine_det_score_thres:
                    continue
                dict_json[i]['annotations'].append({'bbox': dets[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i]['label'][j], 'src': 'det', 'score': dets[i]['score'][j]})
                dict_json[i]['det_count'] += 1

    for f in sot_filelist:
        print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
        with gzip.open(f, 'rt') as fp:
            _t = json.loads(fp.read())
            _forward, _backward = _t['forward'], _t['backward']
        assert len(_forward) == len(dict_json) and len(_backward) == len(dict_json), 'tracking & dataset mismatch'
        for i in range(0, len(_forward)):
            for tr in _forward[i]:
                dict_json[i]['annotations'].append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
                dict_json[i]['sot_count'] += 1
        for i in range(0, len(_backward)):
            for tr in _backward[i]:
                dict_json[i]['annotations'].append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
                dict_json[i]['sot_count'] += 1
    print('finish reading from detection & tracking results')

    if args.refine_remove_no_sot:
        dict_json = list(filter(lambda x: x['sot_count'] > 1, dict_json))
        print('removed all images without tracking results')

    if args.fn_max_samples > 0:
        print('%s [%.2fMB]' % (fn_file, os.path.getsize(fn_file) / (1024 ** 2)))
        with gzip.open(fn_file, 'rt') as fp:
            dets = json.loads(fp.read())['dets']
        for m in args.anno_models:
            assert len(dets[m]) == len(dict_json)
        for i in range(0, len(dets[args.anno_models[0]])):
            fn_annotations = []
            for m in args.anno_models:
                for j in range(0, len(dets[m][i]['label'])):
                    if dets[m][i]['obj_score'][j] < args.fn_min_score: continue
                    x1, y1, x2, y2 = dets[m][i]['bbox'][j]
                    if x2 - x1 > args.fn_max_width_p * dict_json[i]['width']: continue
                    if y2 - y1 > args.fn_max_height_p * dict_json[i]['height']: continue
                    if (x2 - x1) * (y2 - y1) < args.fn_min_area: continue
                    fn_annotations.append({'bbox': dets[m][i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[m][i]['label'][j], 'src': 'fn', 'obj_score': dets[m][i]['obj_score'][j]})
            if len(fn_annotations) > min(args.fn_max_samples, args.fn_max_samples_det_p * dict_json[i]['det_count']):
                random.shuffle(fn_annotations)
                fn_annotations = fn_annotations[:args.fn_max_samples]
            dict_json[i]['annotations'] = dict_json[i]['annotations'] + fn_annotations
            dict_json[i]['fn_count'] = len(fn_annotations)
        print('finish reading from hard negative mining results')

    # dict_json = dict_json[:len(dict_json) // 20]
    # min_bbox_width, max_bbox_width = 5, 1000
    count_all = {'all': 0, 'det': 0, 'sot': 0, 'fn': 0}
    for annotations in dict_json:
        count_all['det'] += annotations['det_count']
        count_all['sot'] += annotations['sot_count']
        count_all['fn'] += annotations['fn_count']
        count_all['all'] += len(annotations['annotations'])
        # annotations['annotations'] = list(filter(lambda ann: min(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) >= min_bbox_width, annotations['annotations']))
        # annotations['annotations'] = list(filter(lambda ann: max(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) <= max_bbox_width, annotations['annotations']))
    print('pseudo annotations: detection %d, tracking %d, fn mining %d, total %d' % (count_all['det'], count_all['sot'], count_all['fn'], count_all['all']))

    pool = ProcessPool(processes=6)
    params_list, chunksize, i = [], len(dict_json) // 20, 0
    while True:
        dict_json_chunk = dict_json[i * chunksize : (i + 1) * chunksize]
        if len(dict_json_chunk) < 1:
            break
        params_list.append({'dict': dict_json_chunk, 'args': args})
        i += 1
    for i in range(0, len(params_list)):
        params_list[i]['desc'] = '%02d/%02d' % (i + 1, len(params_list))
    refine_results = pool.map_async(_graph_refine, params_list).get()
    pool.close()
    pool.join()
    dict_json, count_bboxes = [], 0
    for r in refine_results:
        dict_json = dict_json + r[0]
        count_bboxes += r[1]
    print('%d images, refine bboxes %d => %d' % (len(dict_json), count_all['all'], count_bboxes))

    if visualize:
        assert not args.refine_remove_no_sot, 'cannot visualize with args.refine_remove_no_sot=False'
        loader = torch.utils.data.DataLoader(RefineVisualizeDataset(args.id, dict_json, args.anno_models), batch_size=max(1, args.refine_visualize_workers), shuffle=False, num_workers=args.refine_visualize_workers)
        output_video = os.path.join(dst.lmdb_path, 'refine_%s.mp4' % '_'.join(args.anno_models))
        # writer = skvideo.io.FFmpegWriter(output_video, inputdict={'-r': str(dst.meta['sample_fps'])}, outputdict={'-vcodec': 'libx265', '-r': str(dst.meta['sample_fps']), '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
        writer = skvideo.io.FFmpegWriter(output_video, inputdict={'-r': str(dst.meta['sample_fps'])}, outputdict={'-vcodec': 'hevc_nvenc', '-r': str(dst.meta['sample_fps']), '-pix_fmt': 'yuv420p', '-preset': 'medium', '-rc': 'vbr', '-cq': '25'})
        for f in tqdm.tqdm(loader, ascii=True, desc='writing video'):
            f = f.numpy()
            for j in range(0, f.shape[0]):
                writer.writeFrame(f[j])
        writer.close()
        print('refined video saved to:', output_video)

    return dict_json, count_bboxes


def all_pseudo_annotations(args):
    random.seed(42)
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    dict_json_all, count_bboxes_all, id_back = [], 0, args.id
    for v in video_id_list:
        args.id = v
        dict_json_v, count_bboxes_v = refine_annotations(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            count_bboxes_v *= images_per_video_cap / len(dict_json_v)
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])
        dict_json_all.append(dict_json_v)
        count_bboxes_all += count_bboxes_v
    args.id = id_back
    print('all videos %d images, %d refine bboxes' % (sum(map(len, dict_json_all)), count_bboxes_all))
    return dict_json_all, count_bboxes_all


def get_annotation_dict(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    return annotations


def all_annotation_dict(args):
    annotations_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        annotations_all = annotations_all + get_annotation_dict(args)
    args.id = id_back
    for i in range(0, len(annotations_all)):
        annotations_all[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(annotations_all), sum(list(map(lambda x: len(x['annotations']), annotations_all)))))
    return annotations_all


######################################################
#####   many RCNN library methods are modified   #####
##### modded RCNN only tested on detectron2 v0.6 #####
#####  with models: R50-FPN, R101-FPN, X101-FPN  #####
######################################################

# wrap detectron2/detectron2/data/dataset_mapper.py:DatasetMapper
# include score information in the mapped dataset, later be used for adjusting per-bbox weights
class DatasetMapperFinetune(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        ret = super(DatasetMapperFinetune, self).__call__(dataset_dict)
        score_theta, score_lambda = 0.5, 0.7
        _, y_scale, x_scale = ret['image'].size()
        ret['instances'].x_scale = torch.tensor([x_scale] * ret['instances'].gt_classes.size(0))
        ret['instances'].y_scale = torch.tensor([y_scale] * ret['instances'].gt_classes.size(0))
        if 'annotations' in dataset_dict:
            pseudo_scores = [1.0 for _ in range(0, ret['instances'].gt_classes.size(0))]
            # not sure why this happens, but it seems very rare, just use all 1.0 for pseudo_scores
            if len(dataset_dict['annotations']) != len(pseudo_scores):
                print('gt_classes & annotations mismatch %s %s' % (len(ret['instances'].gt_classes), len(dataset_dict['annotations'])))
                # print(ret['instances'], dataset_dict)
                ret['instances'].annotation_type = torch.tensor([AnnotationType.RELIABLE for _ in range(0, len(pseudo_scores))], dtype=torch.int8)
                ret['instances'].pseudo_scores = torch.tensor(pseudo_scores)
            else:
                annotation_type = [AnnotationType.PSEUDO if 'src' in ann else AnnotationType.RELIABLE for ann in dataset_dict['annotations']]
                annotation_type = torch.tensor(annotation_type, dtype=torch.int8)
                assert annotation_type.sum() == 0 or annotation_type.sum() == annotation_type.size(0), str(dataset_dict['annotations']) # should not have mixture of sources
                # all fields in an instance must have the same length
                if annotation_type.sum() != 0:
                    for i in range(0, len(pseudo_scores)):
                        ann = dataset_dict['annotations'][i]
                        if ann['src'] == 'det':
                            pseudo_scores[i] = ann['score'] * score_lambda + (1 - score_lambda) * 1.0
                        elif ann['src'] == 'sot':
                            # pseudo_scores[i] = ann['init_score'] - ann['track_length'] * 0.025
                            pseudo_scores[i] = score_theta * score_lambda + (1 - score_lambda) * 1.0
                        elif ann['src'] == 'fn':
                            pseudo_scores[i] = ann['obj_score'] * score_lambda + (1 - score_lambda) * 1.0
                        else:
                            raise NotImplementedError # this should not happen
                ret['instances'].annotation_type = annotation_type
                ret['instances'].pseudo_scores = torch.tensor(pseudo_scores)
        return ret
    @staticmethod
    def create_from_sup(mapper):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperFinetune
        return mapper


# DefaultTrainer._trainer is instance of SimpleTrainer
# DefaultTrainer & SimpleTrainer are subclass of TrainerBase
def finetune_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start

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


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneTrainer(DefaultTrainer):
    def __init__(self, cfg, gmm_models=None):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        if cfg.CKPT is not None:
            try:
                state_dict = torch.load(cfg.CKPT)['model']
            except:
                state_dict = torch.load(cfg.CKPT)
            # model.load_state_dict(convert_base_to_moe(state_dict, cfg), strict=True)
            model.load_state_dict(state_dict, strict=True)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []

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
            # mapper = {
            #     'budget': model.budget,
            #     'video_id_to_index': model.video_id_to_index,
            #     'used_indices': model.used_indices,
            #     'un_used_indices': model.un_used_indices
            # }
            # print(mapper)
            # torch.save(mapper, prefix + '.mapper.pth')
            print('saved model state to:', prefix)
        ret.append(detectron2.engine.hooks.EvalHook(self.cfg.SOLVER.SAVE_INTERVAL, save_model_state))        
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=finetune_output)

    @classmethod
    def build_optimizer(cls, cfg, model):
        plain_model = model.model
        args = cfg_to_args(cfg)
        param_dicts = get_param_dict(args, plain_model)
        
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.BASE_LR,
                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return optimizer

def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    _args.smallscale = False
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(_args, 'valid')
    if args.not_eval_coco:
        print('use dummy MSCOCO2017-validation during training')
        dst_cocovalid = dst_cocovalid[:5] + dst_cocovalid[-5:]

    if args.id in video_id_list:
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, get_annotation_dict(args)
        desc_pseudo_anno = 'refine_' + '_'.join(args.anno_models)
        dst_pseudo_anno = refine_annotations(args)[0]
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(_args, 'train')
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
    elif args.id == 'compound':
        import functools
        args.id = '_compound'
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, all_annotation_dict(args)
        desc_pseudo_anno = 'refine_' + '_'.join(args.anno_models)
        dst_pseudo_anno = all_pseudo_annotations(args)[0]
        dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(_args, 'train')
            dst_cocotrain = dst_cocotrain * (len(dst_pseudo_anno) // len(dst_cocotrain) + 1)
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
    else:
        raise NotImplementedError

    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    if args.ckpt is not None and os.access(args.ckpt, os.R_OK):
        print('loading checkpoint:', args.ckpt)
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    else:
        cfg = get_cfg_base_model(args.model)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output

    # disable random flipping & cropping
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.INPUT.CROP.ENABLED = False

    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid, desc_cocovalid)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = FinetuneTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    from rcnn_mod import GeneralizedRCNNFinetune
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperFinetune.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj)
    assert isinstance(trainer.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
    trainer.model = GeneralizedRCNNFinetune.create_from_sup(trainer.model)
    trainer.resume_or_load(resume=False)

    prefix = 'adapt%s_%s_anno_%s%s' % (args.id, args.model, desc_pseudo_anno, '' if args.fn_max_samples <= 0 else '_fn%.4f_%d' % (args.fn_min_score, args.fn_max_samples))
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    with open(os.path.join(args.outputdir, prefix + '.json'), 'r') as fp:
        data = json.load(fp)
    aps, lr_history, loss_history = data['results'], data['lr_history'], data['loss_history']
    iter_list = sorted(list(map(int, aps.keys())))
    dst_list = [desc_cocovalid, desc_manual_valid]
    assert len(dst_list) == 2
    dst_list = {k: {'mAP': [], 'AP50': []} for k in dst_list}
    for i in iter_list:
        for k in dst_list:
            dst_list[k]['mAP'].append(aps[str(i)][k]['bbox']['AP'])
            dst_list[k]['AP50'].append(aps[str(i)][k]['bbox']['AP50'])

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
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid]['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid]['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP', 'Manual Valid AP50', 'Manual Valid mAP'])
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
    exit(0)


class EvaluationDataset(torchdata.Dataset):
    def __init__(self, image_dicts, image_list):
        super(EvaluationDataset, self).__init__()
        assert len(image_dicts) == len(image_list)
        self.image_dicts = image_dicts
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, i):
        im = skimage.io.imread(self.image_list[i])
        if len(im.shape) == 2:
            im = np.stack([im] * 3, axis=2)
        return self.image_dicts[i], im[:, :, ::-1]
    @staticmethod
    def collate(batch):
        return batch


def evaluate(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = DefaultPredictor(cfg)

    results = {}
    detections = []
    loader = torchdata.DataLoader(
        EvaluationDataset(
            copy.deepcopy(images),
            [os.path.join(inputdir, 'unmasked', im['file_name']) for im in images]
        ),
        batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=4
    )
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting %s validation frames' % args.id):
        det = copy.deepcopy(im)
        det['annotations'] = []
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results['manual_%s' % args.id] = evaluate_masked(args.id, detections, outputfile=args.eval_outputfile)

    if not args.eval_skip_coco:
        args.smallscale = False
        detections = get_coco_dicts(args, 'valid')
        for im in tqdm.tqdm(detections, ascii=True, desc='detecting MSCOCO2017 valid'):
            im_arr = skimage.io.imread(im['file_name'])
            if len(im_arr.shape) == 2:
                im_arr = np.stack([im_arr] * 3, axis=2)
            instances = detector(im_arr[:, :, ::-1])['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            im['annotations'] = []
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results['mscoco2017_valid'] = evaluate_cocovalid(args.cocodir, detections)

    print(vars(args))
    for dst in results:
        print('\n            %s\n' % dst)
        print(   '             %s' % '/'.join(results[dst]['metrics']))
        for c in sorted(results[dst]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[dst]['results'][c])))
    return vars(args), results


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = DefaultPredictor(cfg)
    images_tensor = []
    for im in images:
        im_arr = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
        tf = detector.aug.get_transform(im_arr)
        images_tensor.append(torch.as_tensor(tf.apply_image(im_arr).astype('float32').transpose(2, 0, 1)))
    N1, N2 = 100, 900
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            detector.model.inference([{'image': images_tensor[i % len(images)], 'height': images[i % len(images)]['height'], 'width': images[i % len(images)]['width']}])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')

    parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

    parser.add_argument('--fn_min_score', type=float, default=0.99, help='minimum objectiveness score of false negatives')
    parser.add_argument('--fn_max_samples', type=int, default=-1, help='maximum number of false negatives per frame')
    parser.add_argument('--fn_max_samples_det_p', type=float, default=0.5, help='maximum number of false negatives per frame as percentage of number of detections')
    parser.add_argument('--fn_min_area', type=float, default=50, help='minimum area of false negative boxes')
    parser.add_argument('--fn_max_width_p', type=float, default=0.3333, help='maximum percentage width of false negative boxes')
    parser.add_argument('--fn_max_height_p', type=float, default=0.3333, help='maximum percentage height of false negative boxes')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--refine_visualize_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    parser.add_argument('--eval_outputfile', default=None, type=str)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    args.anno_models = sorted(list(set(args.anno_models)))
    assert len(args.anno_models) > 0
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'adapt':
        adapt(args)
    elif args.opt == 'refine':
        refine_annotations(args, visualize=True)
    elif args.opt == 'eval':
        evaluate(args)
    elif args.opt == 'tp':
        inference_throughput(args)
    else:
        pass
    exit(0)


'''
conda deactivate && conda activate detectron2
cd /nfs/detection/zekun/Intersections/scripts/baseline

python finetune.py --id 001 --opt refine --anno_models r50-fpn-3x
python finetune.py --id 001 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r50-c4-3x r101-fpn-3x x101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 2 --iters 25000 --eval_interval 1500
python finetune.py --opt eval --id 001 --cocodir ../../../MSCOCO2017 --model r50-fpn-3x --ckpt adapt_r50-fpn-3x_anno_refine_r50-fpn-3x.pth
python finetune.py --opt tp --id 001 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --anno_models r101-fpn-3x

python finetune.py --id 001 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --train_on_coco 1 --image_batch_size 4

python finetune.py --opt adapt --id compound --model r101-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 200000 --eval_interval 18000 --image_batch_size 4 --num_workers 4 --hold 18

tests

python finetune.py --id 050 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 0 --iters 300 --eval_interval 30 --train_on_coco 1 --image_batch_size 2 --not_eval_coco 1 --lr 0.01 --not_use_mod_rcnn 1
python finetune.py --id 050 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 0 --iters 300 --eval_interval 30 --train_on_coco 1 --image_batch_size 2 --not_eval_coco 1 --lr 0.01 --gmm_max_samples 7000
'''
