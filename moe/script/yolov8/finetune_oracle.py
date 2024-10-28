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
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict
from multiprocessing import Pool as ProcessPool
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import functools
import imantics
import contextlib

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
from detectron2.config import get_cfg

import logging
import weakref

# Adding parent directory to sys.path for importing custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base_detector_train import get_coco_dicts
from finetune import get_annotation_dict, FinetuneTrainer, finetune_simple_trainer_run_step
from evaluation import eval_AP, _check_overlap
from yolov8 import *
from inference_server_simulate_yolov8 import YOLOServer

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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_oracle_output')


def get_oracle_splits(args):
    """Generate training and validation splits for oracle evaluation."""
    train_per_video = 400
    images_train, images_valid = [], []
    for args.id in video_id_list:
        images = get_annotation_dict(args)
        images = sorted(images, key=lambda x: x['file_name'])
        for im in images:
            im['video_id'] = args.id
            im['file_name'] = os.path.basename(im['file_name'])
        train_N = min(max(1, math.floor(args.train_r * len(images))), len(images) - 1)
        images_train_v, images_valid_v = images[:train_N], images[train_N:]
        assert len(images_train_v) > 0 and len(images_valid_v) > 0
        images_valid.extend(copy.deepcopy(images_valid_v))
        assert len(images_train_v) < train_per_video
        images_train_v = images_train_v * (train_per_video * 2 // len(images_train_v))
        random.shuffle(images_train_v)
        images_train_v = images_train_v[:train_per_video]
        images_train.extend(copy.deepcopy(images_train_v))

    print('oracle training set:   %d images, %d bboxes' % (len(images_train), sum(list(map(lambda x: len(x['annotations']), images_train)))))
    print('oracle validation set: %d images, %d bboxes' % (len(images_valid), sum(list(map(lambda x: len(x['annotations']), images_valid)))))
    return images_train, images_valid


def adapt(args):
    """Adapt the model using oracle training and validation splits."""
    random.seed(42)
    desc_train = f'oracle_train{args.train_r:.2f}_cocotrain'
    desc_valid = f'oracle_{1 - args.train_r:.2f}valid'
    images_train, images_valid = get_oracle_splits(args)
    images_base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated'))

    for im in images_train:
        im['file_name'] = os.path.join(images_base_dir, im['video_id'], 'unmasked', im['file_name'])
    for im in images_valid:
        im['file_name'] = os.path.join(images_base_dir, im['video_id'], 'masked', im['file_name'])

    dst_cocotrain = get_coco_dicts(args, 'train')
    random.shuffle(dst_cocotrain)
    images_train = images_train + dst_cocotrain[: len(images_train) // 5]

    for i, im in enumerate(images_train):
        im['image_id'] = i + 1
    for i, im in enumerate(images_valid):
        im['image_id'] = i + 1
    print('include MSCOCO2017 training images, totally %d images' % len(images_train))

    DatasetCatalog.register(desc_train, lambda: images_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid, lambda: images_valid)
    MetadataCatalog.get(desc_valid).thing_classes = thing_classes

    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK)
    print('loading checkpoint:', args.ckpt)
    cfg.MODEL.WEIGHTS = args.ckpt
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 20
    cfg.SOLVER.GAMMA = 1
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.SAVE_PREFIX = os.path.join(args.outputdir, "finetune_oracle")
    cfg.SOLVER.SAVE_INTERVAL = args.save_interval
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_train,)
    cfg.DATASETS.TEST = (desc_valid,)
    cfg.YOLO_CONFIG_PATH = args.config
    cfg.CKPT = args.ckpt

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = FinetuneTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)

    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt_%s_%s' % (args.model, desc_train)
    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    torch.save(trainer.model.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = aps.keys()
    dst_list = {'mAP': [aps[i]['bbox']['AP'] for i in iter_list], 'AP50': [aps[i]['bbox']['AP50'] for i in iter_list]}

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L: i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1:, :]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Manual Valid AP50', 'Manual Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')
    plt.subplot(1, 2, 2)
    colors = ['#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#000000']
    color_i = 0
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
    plt.close()


class EvaluationDataset(torchdata.Dataset):
    """Dataset class for evaluation."""

    def __init__(self, cfg, images):
        super(EvaluationDataset, self).__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == 'BGR'
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        """Read and preprocess an image for evaluation."""
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        return {'image': image, 'height': height, 'width': width, 'video_id': self.images[i]['video_id']}, self.images[i]

    @staticmethod
    def collate(batch):
        """Custom collation function for DataLoader."""
        return batch


def evaluate_all_videos(args):
    """Evaluate the model on all videos."""
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

    _, images_valid = get_oracle_splits(args)
    images_base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated'))

    for im in images_valid:
        assert 'video_id' in im
        im['file_name'] = os.path.join(images_base_dir, im['video_id'], 'unmasked', im['file_name'])

    loader = torchdata.DataLoader(
        EvaluationDataset(cfg, images_valid),
        batch_size=None,
        collate_fn=EvaluationDataset.collate,
        shuffle=False,
        num_workers=args.num_workers
    )

    detections = {v: [] for v in video_id_list}
    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(images_valid), desc='detecting'):
        det = copy.deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            instances = model.inference([inputs])[0]['instances'].to('cpu')
            det['instances'] = {
                'bbox': instances.pred_boxes.tensor,
                'score': instances.scores,
                'label': instances.pred_classes
            }
        detections[im['video_id']].append((im, det))

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
    for cat in ['person', 'vehicle', 'overall', 'weighted']:
        APs = np.array([results[video_id]['results'][cat] for video_id in results]) * 100
        print('%s: mAP %.2f, AP50 %.2f' % (cat, APs[:, 0].mean(), APs[:, 1].mean()))


def _run_AP_eval(detections, check_empty=True):
    """Run AP evaluation on detections."""
    results = {}
    for video_id in detections:
        with open(os.path.join(os.path.dirname(__file__), '..', '..', 'masks.json'), 'r') as fp:
            mask = json.load(fp)
        mask = {m['video']: m['polygons'] for m in mask}
        mask = mask[video_id]
        im_arr = skimage.io.imread(detections[video_id][0][0]['file_name'])
        if len(mask) > 0:
            m_arr = imantics.Annotation.from_polygons(mask, image=imantics.Image(im_arr))
            m_arr = np.expand_dims(m_arr.array.astype(np.float16), 2) * 0.5
        else:
            m_arr = None

        # if check_empty and video_id == '152':
        #     # breakpoint()
        #     for im in detections[video_id][1]:
        #         if len(im['annotations']) == 0:
        #             print("insert random boxes.")
        #             x1, x2 = sorted((np.random.rand(2) * im['width'] * 0.8 + 2).tolist())
        #             y1, y2 = sorted((np.random.rand(2) * im['height'] * 0.8 + 2).tolist())
        #             im['annotations'].append({
        #                 'bbox': list(map(float, [x1, y1, x2, y2])), 
        #                 'bbox_mode': BoxMode.XYXY_ABS, 
        #                 'segmentation': [], 
        #                 'category_id': 0, 
        #                 'score': 0.5
        #             })
        
        images_v, detections_v = [], []
        for im, det in detections[video_id]:
            bbox, score, label = det['instances']['bbox'].numpy().tolist(), det['instances']['score'].numpy().tolist(), det['instances']['label'].numpy().tolist()
            for i in range(0, len(label)):
                if not _check_overlap(m_arr, bbox[i]):
                    det['annotations'].append({
                        'bbox': bbox[i],
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'segmentation': [],
                        'category_id': label[i],
                        'score': score[i]
                    })
            del det['instances']
            detections_v.append(det)
            images_v.append(im)
        assert len(images_v) == len(detections_v)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = eval_AP(images_v, detections_v, return_thres=False)
        del results[video_id]['raw']
        print(video_id, end=' ', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--model', type=str, default='yolov8s')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--mapper', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--config', type=str, default='../../configs/yolov8s.yaml', help='detection model config path')
    parser.add_argument('--budget', type=int)

    parser.add_argument('--cocodir', type=str, default='../../MSCOCO2017')
    parser.add_argument('--smallscale', type=bool, default=False)
    parser.add_argument('--train_r', type=float, default=0.5)
    parser.add_argument('--eval_moe', type=bool, default=False)
    parser.add_argument('--split_list', type=int, nargs='+')

    parser.add_argument('--iters', type=int, default=1200)
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    print(args)

    finetune_output = args.outputdir
    if not os.access(finetune_output, os.W_OK):
        os.makedirs(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.access(args.outputdir, os.W_OK)
    assert 0.1 <= args.train_r <= 0.9

    if args.opt == 'adapt':
        adapt(args)
    elif args.opt == 'eval':
        evaluate_all_videos(args)


'''
python finetune_oracle.py --opt adapt --train_r 0.5 --ckpt ../../models/mscoco2017_remap_r18-fpn-3x.pth --cocodir ../../../MSCOCO2017 --iters 1000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --outputdir /tmp

python finetune_oracle.py --opt eval --train_r 0.5  --num_workers 4 --ckpt ../../models/yolov8s_remap.pth
python finetune_oracle.py --opt eval --eval_moe 1 --train_r 0.5 --num_workers 4 --ckpt /mnt/f/intersections_results/cvpr24/paper_models/r18.budget10.cont.iter.80000

'''