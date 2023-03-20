#!python3

import os
import sys
# import json
# import copy
import random
import tqdm
import argparse
from collections import deque
from multiprocessing import Pool as ProcessPool
import numpy as np
import imageio

import cv2
from detectron2.structures import BoxMode

sys.path.append(os.path.join(os.path.dirname(__file__)))
from adaptation.constants import video_id_list
from adaptation.scenes100_dataset import refine_pseudo_labels


def extract(args):
    assert os.path.isdir(args.outputdir) and os.access(args.outputdir, os.W_OK), 'output directory not writable: ' + str(args.outputdir)
    backgrounddir, inpaintdir = os.path.join(args.outputdir, 'pngs'), os.path.join(args.outputdir, 'inpaint')
    if not os.access(backgrounddir, os.W_OK):
        os.mkdir(backgrounddir)
    if not os.access(inpaintdir, os.W_OK):
        os.mkdir(inpaintdir)

    random.seed(42)
    # training images with pseudo-labeling
    args.fusion = 'vanilla'
    images = refine_pseudo_labels(args)
    images.sort(key=lambda x: x['file_name'])

    sample_idx = set(np.arange(0, len(images), args.sample_interval * args.fps).astype(int).tolist())
    background_idx = set(np.arange(0, len(images), args.background_interval * args.fps).astype(int).tolist()[1:])
    buffer_size = 100
    print('input FPS=%.1f, sample every %.1f sec total %d images, background every %.1f sec total %d images' % (args.fps, args.sample_interval, len(sample_idx), args.background_interval, len(background_idx)))

    Q = deque([], maxlen=buffer_size)
    background_filenames = []
    for i in tqdm.tqdm(range(0, len(images)), ascii=True, desc='extracting background'):
        if i in sample_idx:
            im_arr = np.asarray(imageio.v2.imread(images[i]['file_name']))
            anns = images[i]['annotations']
            M_arr = np.ones_like(im_arr[:, :, 0 : 1])
            for ann in anns:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                x1, y1, x2, y2 = map(int, map(lambda x: max(x, 0), ann['bbox']))
                M_arr[y1 : y2, x1 : x2] = 0.0
            Q.append({'im_arr': im_arr, 'M_arr': M_arr, 'fn': os.path.basename(images[i]['file_name']), 'annotations': anns})

        if i in background_idx:
            images_arr = np.stack(list(map(lambda x: x['im_arr'], Q))).astype(np.float16)
            masks_arr = np.stack(list(map(lambda x: x['M_arr'], Q))).astype(np.float16)
            average, images_arr, masks_arr = images_arr.mean(axis=0), (images_arr * masks_arr).mean(axis=0), masks_arr.mean(axis=0)
            M = np.zeros(shape=masks_arr.shape, dtype=np.uint8) + 255
            for x in range(0, images_arr.shape[0]):
                for y in range(0, images_arr.shape[1]):
                    if masks_arr[x, y, 0] < 0.5 / len(Q):
                        masks_arr[x, y], images_arr[x, y], M[x, y] = 1, average[x, y], 0
            f1 = os.path.join(backgrounddir, '%s_mask.png' % Q[-1]['fn'])
            f2 = os.path.join(backgrounddir, '%s_background.jpg' % Q[-1]['fn'])
            imageio.imwrite(f1, M)
            imageio.imwrite(f2, (images_arr / masks_arr).astype(np.uint8), quality=80)
            background_filenames.append([Q[-1]['fn'], f1, f2])

    params_list = []
    for fn, f1, f2 in background_filenames:
        im_M = imageio.v2.imread(f1)
        im_bg = imageio.v2.imread(f2)
        params_list.append([im_bg, im_M, os.path.join(inpaintdir, fn + '_inpaint.jpg'), {'quality': 80}])
    print('inpainting background holes')
    pool = ProcessPool(processes=args.procs)
    _ = pool.map_async(_inpaint_background, params_list).get()
    pool.close()
    pool.join()


def _inpaint_background(param):
    im_bg, im_M, fn, imsave_params = param
    assert im_bg.dtype == im_M.dtype and im_bg.shape[:2] == im_M.shape
    im_M = 255 - im_M
    pixels = im_M.sum() / 255.0
    if pixels < 10:
        pass
    else:
        R = min(im_M.shape) // 10
        im_bg = cv2.inpaint(im_bg, im_M, R, cv2.INPAINT_TELEA)
    imageio.imwrite(fn, im_bg, **imsave_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Background Extraction Script')
    # generic arguments
    parser.add_argument('--id', type=str, choices=video_id_list, help='video ID')
    parser.add_argument('--fps', type=float, default=5, help='training images framerate')
    parser.add_argument('--sample_interval', type=float, default=2, help='interval in seconds to sample an image')
    parser.add_argument('--background_interval', type=float, default=90, help='interval in seconds to generate a background image')
    parser.add_argument('--procs', type=int, default=8, help='parallel processes for image inpainting')
    parser.add_argument('--outputdir', type=str, default='background', help='save background images to this directory')

    # pseudo-labeling hyper-parameters
    parser.add_argument('--anno_models', nargs='+', default=['r101-fpn-3x', 'r50-fpn-3x'], help='base models used for pseudo-labeling')
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score for pseudo-labeling')
    parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    parser.add_argument('--refine_no_sot', type=bool, default=False, help='do not include tracking bounding boxes')
    args = parser.parse_args()

    args.anno_models = sorted(list(set(args.anno_models)))
    print(args)
    extract(args)
