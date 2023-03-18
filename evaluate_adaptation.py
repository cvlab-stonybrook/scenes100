#!python3

import os
import sys
import json
import copy
import tqdm
import glob
import argparse
import numpy as np
import contextlib
import matplotlib.pyplot as plt

import torch
import detectron2
from detectron2.structures import BoxMode

sys.path.append(os.path.join(os.path.dirname(__file__)))
from adaptation.constants import video_id_list, thing_classes
from adaptation.scenes100_dataset import get_manual_dicts
from adaptation.base_model_cfg import get_cfg_base_model
from adaptation.evaluator import evaluate_masked
from adaptation.predictor import AdaptationPredictor
from adaptation.dataloader import construct_image_w_background


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dicts, image_format):
        super(EvaluationDataset, self).__init__()
        self.image_dicts = image_dicts
        self.image_format = image_format
    def __len__(self):
        return len(self.image_dicts)
    def __getitem__(self, i):
        image = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name'], format=self.image_format)
        if 'file_name_background' in self.image_dicts[i]:
            image_bg = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name_background'], format=self.image_format)
            image, _, image_diff = construct_image_w_background(image, image_bg)
        else:
            image_diff = None
        return {'dict': copy.deepcopy(self.image_dicts[i]), 'image': image, 'image_diff': image_diff}
    @staticmethod
    def collate(batch):
        return batch


def evaluate_base(args):
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.FUSION = 'vanilla'
    predictor = AdaptationPredictor(cfg)
    results_AP, detections_per_video = {}, {}

    for video_id in video_id_list:
        detections_per_video[video_id] = []
        images = get_manual_dicts(video_id)
        loader = torch.utils.data.DataLoader(
            EvaluationDataset(images, cfg.INPUT.FORMAT),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=2
        )

        for batch in tqdm.tqdm(loader, total=len(images), ascii=True):
            instances = predictor(batch['image'], batch['image_diff'])[0]['instances'].to('cpu')
            im = batch['dict']
            im['annotations'] = []
            im['instances'] = instances
            detections_per_video[video_id].append(im)
        for im in detections_per_video[video_id]:
            # bbox has format [x1, y1, x2, y2]
            bbox = im['instances'].pred_boxes.tensor
            score = im['instances'].scores
            label = im['instances'].pred_classes
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
            del im['instances']

        # AP evaluation
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results_AP[video_id] = evaluate_masked(video_id, copy.deepcopy(detections_per_video[video_id]))
        print(video_id, results_AP[video_id]['results'])
    with open(args.base_result_json, 'w') as fp:
        json.dump({'AP': results_AP, 'detection': detections_per_video}, fp)

    categories = thing_classes + ['overall', 'weighted']
    xs = np.arange(0, len(results_AP), 1)
    _, axes = plt.subplots(4, 1, figsize=(28, 28))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        mAP_AP50 = np.array([results_AP[v]['results'][categories[i]] for v in results_AP]) * 100
        valid_mask = mAP_AP50[:, 0] >= 0
        axes[i].plot(xs[valid_mask], mAP_AP50[valid_mask, 0], 'rx-')
        axes[i].plot(xs[valid_mask], mAP_AP50[valid_mask, 1], 'bx-')
        axes[i].legend([
            'mAP valid mean: %.4f' % mAP_AP50[valid_mask, 0].mean(),
            'AP50 valid mean: %.4f' % mAP_AP50[valid_mask, 1].mean()
        ])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(list(results_AP.keys()), rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        axes[i].set_ylim(0, 105)
        axes[i].set_ylabel('AP (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    plt.suptitle(args.base_result_json)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(args.base_result_json[: -4] + 'pdf')
    plt.close()


def evaluate_single(args):
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), 'checkpoint not readable: ' + args.ckpt
    cfg.FUSION = args.fusion
    predictor = AdaptationPredictor(cfg)
    images = get_manual_dicts(args.id)
    if 'fusion' in args.fusion:
        if args.eval_background == 'last':
            background_file_last = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100', 'train_background', args.id, 'inpaint', '*inpaint.jpg'))))[-1]
            for im in images:
                im['file_name_background'] = background_file_last
        elif args.eval_background == 'dynamic':
            for im in images:
                im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100', 'valid_background', args.id, 'inpaint', os.path.basename(im['file_name']) + '_inpaint.jpg'))
        else:
            raise NotImplementedError
    loader = torch.utils.data.DataLoader(
        EvaluationDataset(images, cfg.INPUT.FORMAT),
        batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=2
    )

    detections = []
    for batch in tqdm.tqdm(loader, total=len(images), ascii=True):
        instances = predictor(batch['image'], batch['image_diff'])[0]['instances'].to('cpu')
        im = batch['dict']
        im['annotations'] = []
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor
        score = instances.scores
        label = instances.pred_classes
        for i in range(0, len(label)):
            im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
        detections.append(im)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results_AP = evaluate_masked(args.id, copy.deepcopy(detections))
    print(results_AP)
    return results_AP


def evaluate_batch(args):
    from adaptation.constants import video_id_list
    print('scanning checkpoints in %s' % args.compare_ckpts_dir)
    ckpt_dict = {}
    for ckpt in sorted(glob.glob(os.path.join(args.compare_ckpts_dir, 'adapt*.pth'))):
        video_id = os.path.basename(ckpt)[5 : 8]
        if not video_id in ckpt_dict:
            ckpt_dict[video_id] = []
        ckpt_dict[video_id].append(ckpt)
    for video_id in ckpt_dict:
        assert video_id in video_id_list, 'unrecognizable video ID: ' + video_id
        assert len(ckpt_dict[video_id]) == 1, 'more than 1 checkpoints for video ' + video_id + ': ' + str(ckpt_dict[video_id])
    results_AP = {}
    for i, video_id in enumerate(ckpt_dict):
        args.id, args.ckpt = video_id, ckpt_dict[video_id][0]
        results_AP[video_id] = evaluate_single(args)
        print('%d/%d finished\n' % (i + 1, len(ckpt_dict)))
    with open(os.path.join(args.compare_ckpts_dir, 'results_compare.json'), 'w') as fp:
        json.dump(results_AP, fp)
    
    with open(args.base_result_json, 'r') as fp:
        results_AP_base = json.load(fp)['AP']
    videos = sorted(list(results_AP.keys()))
    categories = thing_classes + ['overall', 'weighted']

    improvements = {c: [] for c in categories}
    for video_id in videos:
        AP1 = results_AP_base[video_id]['results']
        AP2 = results_AP[video_id]['results']
        for cat in categories:
            improvements[cat].append([AP2[cat][0] - AP1[cat][0], AP2[cat][1] - AP1[cat][1]])
    for cat in categories:
        improvements[cat] = np.array(improvements[cat]) * 100.0

    xs = np.arange(0, len(videos), 1)
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        axes[i].plot([-1, xs.max() + 1], [0, 0], 'k-')
        axes[i].plot(xs, improvements[categories[i]][:, 0], 'r.-')
        axes[i].plot(xs, improvements[categories[i]][:, 1], 'b.-')
        axes[i].legend(['0', 'mAP %.4f' % improvements[categories[i]][:, 0].mean(), 'AP50 %.4f' % improvements[categories[i]][:, 1].mean()])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(videos, rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        axes[i].set_ylabel('AP improvement (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    plt.suptitle(args.compare_ckpts_dir)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(args.compare_ckpts_dir, 'results_compare.pdf'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptation Evaluation Script')
    # generic arguments
    parser.add_argument('--opt', type=str, choices=['base', 'single', 'batch'], help='evaluate base model, single adapted model, or batch of adapted models')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+[''], help='video ID')
    parser.add_argument('--model', type=str, choices=['r50-fpn-3x', 'r101-fpn-3x'], help='detection model')
    parser.add_argument('--ckpt', type=str, help='weights checkpoint of model')
    parser.add_argument('--compare_ckpts_dir', type=str, help='directory of adapted models')
    parser.add_argument('--base_result_json', type=str, help='JSON file that used to save the evaluation results for the base models')

    # object mask fusion options
    parser.add_argument('--fusion', type=str, choices=['vanilla', 'earlyfusion', 'midfusion', 'latefusion'], help='vanilla/early-/mid-/late- fusion')
    parser.add_argument('--eval_background', type=str, default='last', choices=['dynamic', 'last'], help='use inference time dynamic background or last training time background')
    args = parser.parse_args()

    if args.opt == 'base':
        evaluate_base(args)
    elif args.opt == 'single':
        evaluate_single(args)
    elif args.opt == 'batch':
        evaluate_batch(args)

'''
evaluate single model:
python evaluate_adaptation.py --opt single --id 003 --model r101-fpn-3x --ckpt adapt003_r101-fpn-3x_anno_train_003_refine_r101-fpn-3x_r50-fpn-3x.pth --fusion vanilla
python evaluate_adaptation.py --opt single --id 003 --model r101-fpn-3x --ckpt adapt003_r101-fpn-3x_anno_train_003_refine_r101-fpn-3x_r50-fpn-3x_midfusion_mixup.pth --fusion midfusion --eval_background last
python evaluate_adaptation.py --opt single --id 003 --model r101-fpn-3x --ckpt adapt003_r101-fpn-3x_anno_train_003_refine_r101-fpn-3x_r50-fpn-3x_midfusion_mixup.pth --fusion midfusion --eval_background dynamic

evaluate base model:
python evaluate_adaptation.py --opt base --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_r101-fpn-3x.pth --base_result_json results_base_r101-fpn-3x.json

batch compare:
python evaluate_adaptation.py --opt batch --model r101-fpn-3x --compare_ckpts_dir trained_models/best_midfusion_mixup/ --fusion midfusion --base_result_json results_base_r101-fpn-3x.json
'''
