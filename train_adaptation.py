#!python3

import os
import sys
import json
import copy
import random
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import detectron2
from detectron2.evaluation import inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog

sys.path.append(os.path.join(os.path.dirname(__file__)))
from adaptation.constants import video_id_list, thing_classes
from adaptation.mscoco_remap_dataset import get_coco_dicts
from adaptation.scenes100_dataset import refine_pseudo_labels, get_manual_dicts
from adaptation.trainer import AdaptationTrainer
from adaptation.base_model_cfg import get_cfg_base_model


def adapt(args):
    random.seed(42)
    # training set with pseudo-labeling
    desc_train, dst_train = 'train_%s_refine_%s%s%s' % (args.id, '_'.join(args.anno_models), '' if args.fusion == 'vanilla' else ('_' + args.fusion), '_mixup' if args.mixup else ''), refine_pseudo_labels(args)
    if args.mixup:
        dst_train_copy = copy.deepcopy(dst_train)
        for im in tqdm.tqdm(dst_train, ascii=True, desc='populating mixup sources'):
            im['mixup_src_images'] = [dst_train_copy[random.randrange(0, len(dst_train_copy))]]
        del dst_train_copy

    # validation sets
    desc_manualvalid, dst_manualvalid = 'valid_manual_%s' % args.id, get_manual_dicts(args.id)
    if 'fusion' in args.fusion:
        for im in dst_manualvalid:
            im['file_name_background'] = dst_train[-1]['file_name_background'] # choice of background images here does not affect training
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts('valid', use_background=('fusion' in args.fusion))
    if args.debug:
        dst_cocovalid = dst_cocovalid[: 25] + dst_cocovalid[-25 :]

    # include MSCOCO training images
    dst_cocotrain = get_coco_dicts('train', use_background=('fusion' in args.fusion))
    random.shuffle(dst_cocotrain)
    dst_train = dst_train + dst_cocotrain[: len(dst_train)]
    print('include MSCOCO2017 training images, totally %d images' % len(dst_train))
    for i in range(0, len(dst_train)):
        dst_train[i]['image_id'] = i + 1

    # register datasets
    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_manualvalid, lambda: dst_manualvalid)
    MetadataCatalog.get(desc_manualvalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_train, lambda: dst_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes

    # trainer configuration
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), 'checkpoint not readable: ' + args.ckpt
    cfg.DATALOADER.NUM_WORKERS               = args.num_workers
    cfg.OUTPUT_DIR                           = 'train_output_%s%s' % (args.fusion, '_mixup' if args.mixup else '')
    cfg.SOLVER.IMS_PER_BATCH                 = args.image_batch_size
    cfg.SOLVER.BASE_LR                       = args.lr
    cfg.SOLVER.WARMUP_ITERS                  = args.iters // 10
    cfg.SOLVER.GAMMA                         = 0.5
    cfg.SOLVER.STEPS                         = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER                      = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD                     = args.eval_interval
    cfg.DATASETS.TRAIN                       = (desc_train,)
    cfg.DATASETS.TEST                        = (desc_manualvalid, desc_cocovalid)
    cfg.FUSION                               = args.fusion
    cfg.MULTITASK_LOSS_ALPHA                 = args.multitask_loss_alpha
    cfg.MIXUP                                = args.mixup
    cfg.MIXUP_P                              = args.mixup_p
    cfg.MIXUP_R                              = args.mixup_r
    cfg.MIXUP_OVERLAP_THRES                  = args.mixup_overlap_thres
    cfg.MIXUP_RANDOM_POSITION                = args.mixup_random_position

    trainer = AdaptationTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # run evaluation before any training
    results_0 = {}
    for _, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    # save model and training history
    prefix = 'adapt%s_%s_anno_%s' % (args.id, args.model, desc_train)
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    # visualize training history
    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(list(aps.keys()))
    dst_list = [desc_cocovalid, desc_manualvalid]
    assert len(dst_list) == 2
    dst_list = {k: {'mAP': [], 'AP50': []} for k in dst_list}
    for i in iter_list:
        for k in dst_list:
            dst_list[k]['mAP'].append(aps[i][k]['bbox']['AP'])
            dst_list[k]['AP50'].append(aps[i][k]['bbox']['AP50'])

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
    plt.plot(iter_list, np.array(dst_list[desc_manualvalid]['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_manualvalid]['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP', 'Manual Valid AP50', 'Manual Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.yticks(np.arange(0, 1.01, 0.1))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptation Training Script')
    # generic arguments
    parser.add_argument('--id', type=str, choices=video_id_list, help='video ID')
    parser.add_argument('--model', type=str, choices=['r50-fpn-3x', 'r101-fpn-3x'], help='detection model')
    parser.add_argument('--ckpt', type=str, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.', help='save training results to this directory')

    # pseudo-labeling hyper-parameters
    parser.add_argument('--anno_models', nargs='+', default=[], help='base models used for pseudo-labeling')
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score for pseudo-labeling')
    parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    parser.add_argument('--refine_no_sot', type=bool, default=False, help='do not include tracking bounding boxes')

    # location-aware mixup hyper-parameters
    parser.add_argument('--mixup', type=bool, default=False, help='apply mixup during training')
    parser.add_argument('--mixup_p', type=float, default=0.3, help='probability of applying mixup to an image')
    parser.add_argument('--mixup_r', type=float, default=0.5, help='ratio of mixed-up bounding boxes')
    parser.add_argument('--mixup_overlap_thres', type=float, default=0.65, help='above this threshold, overwritten boxes by mixup are removed')
    parser.add_argument('--mixup_random_position', type=bool, default=False, help='randomly position patch, only used by vanilla models')

    # object mask fusion options
    parser.add_argument('--fusion', type=str, choices=['vanilla', 'earlyfusion', 'midfusion', 'latefusion'], help='vanilla/early-/mid-/late- fusion')
    parser.add_argument('--multitask_loss_alpha', type=float, default=0.5, help='relative weight of 2-branches losses, only used my mid- and late- fusion models')

    # training hyper-parameters
    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--debug', type=bool, default=False, help='use small datasets for quick debugging')
    parser.add_argument('--image_batch_size', default=4, type=int, help='image batch size')
    parser.add_argument('--roi_batch_size', default=128, type=int, help='ROI patch batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='base learning rate')
    parser.add_argument('--num_workers', default=0, type=int, help='number of dataloader processes')
    args = parser.parse_args()

    args.anno_models = sorted(list(set(args.anno_models)))
    assert 0 <= args.multitask_loss_alpha <= 1, str(args.multitask_loss_alpha)
    print(args)
    adapt(args)

'''
# debugging

vanilla:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion vanilla --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2

vanilla w/ mixup:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion vanilla --mixup 1 --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2

early-fusion:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_wdiff_earlyfusion_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion earlyfusion --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2

early-fusion w/ mixup:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_wdiff_earlyfusion_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion earlyfusion --mixup 1 --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2

mid-fusion:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion midfusion --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2

mid-fusion w/ mixup:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion midfusion --mixup 1 --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2

late-fusion:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_wdiff_latefusion_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion latefusion --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2

late-fusion w/ mixup:
python train_adaptation.py --id 003 --model r101-fpn-3x --ckpt mscoco/models/mscoco2017_remap_wdiff_latefusion_r101-fpn-3x.pth --anno_models r101-fpn-3x r50-fpn-3x --fusion latefusion --mixup 1 --iters 200 --eval_interval 101 --debug 1 --image_batch_size 2 --num_workers 2
'''
