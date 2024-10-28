#!python3

import os
import glob
import json
import random
import time
import shutil
import copy
import tqdm
import tempfile

import skimage.io
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imantics

from detectron2.structures import BoxMode
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
mask_rgb, mask_alpha = [0, 1, 0], 0.3


def eval_AP(images, detections, return_thres=False):
    coco_dict = {'info': {'year': 0, 'version': '', 'description': '', 'contributor': '', 'url': '', 'date_created': ''}, 'licenses': [{'id': 0, 'name': '', 'url': ''}], 'categories': None, 'images': None, 'annotations': None}
    categories = []
    for i in range(0, len(thing_classes)):
        categories.append({'id': i, 'name': thing_classes[i], 'supercategory': ''})
    coco_dict['categories'] = categories

    coco_images, coco_annotations, coco_dets = [], [], []
    weight_per_class = [0 for _ in range(0, len(thing_classes))]
    for i in range(0, len(images)):
        coco_images.append({'id': i, 'width': 0, 'height': 0, 'file_name': '', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': ''})
        for ann in images[i]['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            x1, y1, x2, y2 = ann['bbox']
            coco_annotations.append({'id': len(coco_annotations), 'image_id': i, 'category_id': ann['category_id'], 'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'segmentation': [], 'iscrowd': 0})
            weight_per_class[ann['category_id']] += 1
        for ann in detections[i]['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            x1, y1, x2, y2 = ann['bbox']
            coco_dets.append({'id': len(coco_dets), 'image_id': i, 'category_id': ann['category_id'], 'score': ann['score'], 'bbox': [x1, y1, x2 - x1, y2 - y1]})
    coco_dict['images'], coco_dict['annotations'] = coco_images, coco_annotations
    weight_sum = sum(weight_per_class)
    for i in range(0, len(weight_per_class)):
        weight_per_class[i] /= weight_sum

    fd, coco_dict_f = tempfile.mkstemp(suffix='.json', text=True)
    with os.fdopen(fd, 'w') as fp:
        json.dump(coco_dict, fp)
    coco_gt = COCO(coco_dict_f)
    fd, coco_dets_f = tempfile.mkstemp(suffix='.json', text=True)
    with os.fdopen(fd, 'w') as fp:
        json.dump(coco_dets, fp)
    coco_det = coco_gt.loadRes(coco_dets_f)
    os.unlink(coco_dict_f)
    os.unlink(coco_dets_f)

    coco_eval = COCOeval(coco_gt, coco_det, 'bbox')

    print('evaluate for', coco_eval.params.catIds)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {
        'metrics': ['mAP', 'AP50'],
        'thing_classes': thing_classes,
        'results': {
            'overall': coco_eval.stats[:2].tolist()
        },
        'raw': {
            'iouThrs': coco_eval.eval['params'].iouThrs.tolist(),
            'recThrs': coco_eval.eval['params'].recThrs.tolist(),
            'catIds': list(map(int, coco_eval.eval['params'].catIds)) if coco_eval.eval['params'].useCats else [0],
            'areaRng': coco_eval.eval['params'].areaRng,
            'maxDets': coco_eval.eval['params'].maxDets,
            'precision': coco_eval.eval['precision'].tolist(), # (T,R,K,A,M)
            'recall': coco_eval.eval['recall'].tolist(), # (T,K,A,M)
            'scores': coco_eval.eval['scores'].tolist(), # (T,R,K,A,M)
            # T: IoU thres, R: recall thres, K: classes, A: areas, M: max dets
        },
        'weights': {
            'total': weight_sum,
            'classes': weight_per_class
        }
    }

    for i in range(0, len(thing_classes)):
        coco_eval.params.catIds = [i]
        print('evaluate for', coco_eval.params.catIds)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        results['results'][thing_classes[i]] = coco_eval.stats[:2].tolist()

    results['results']['weighted'] = [0.0, 0.0]
    for i in range(0, len(thing_classes)):
        results['results']['weighted'][0] += weight_per_class[i] * results['results'][thing_classes[i]][0]
        results['results']['weighted'][1] += weight_per_class[i] * results['results'][thing_classes[i]][1]

    if return_thres:
        print('evaluate on all category & IoUthres')
        all_thres, all_cats = coco_eval.params.iouThrs[:], [[i] for i in range(0, len(thing_classes))] + [list(range(0, len(thing_classes)))]
        results_all_thres = []
        for cat in all_cats:
            for thres in all_thres:
                coco_eval.params.catIds = cat
                coco_eval.params.iouThrs = [thres]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                results_all_thres.append({'categories': cat, 'thres': thres, 'AP': coco_eval.stats[0]})
        results['all_thres'] = results_all_thres

    return results


def _check_overlap(mask, bbox):
    if mask is None:
        return False
    x1, y1, x2, y2 = map(int, bbox)
    H, W = mask.shape[:2]
    x1 = min(max(x1, 0), W - 1)
    x2 = min(max(x2, 0), W - 1)
    y1 = min(max(y1, 0), H - 1)
    y2 = min(max(y2, 0), H - 1)
    return (mask[y1, x1, 0] > 1e-3 or mask[y1, x2, 0] > 1e-3 or mask[y2, x1, 0] > 1e-3 or mask[y2, x2, 0] > 1e-3)


def evaluate_masked(video_id, detections, outputfile=None, return_bboxes=False, check_empty=False):
    detections = copy.deepcopy(detections)
    print('evaluate %s, output to %s' % (video_id, outputfile))
    inputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotated', video_id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    with open(os.path.join(os.path.dirname(__file__), '..', 'masks.json'), 'r') as fp:
        mask = json.load(fp)
    mask = {m['video']: m['polygons'] for m in mask}
    mask = mask[video_id]

    im_arr = skimage.io.imread(os.path.join(inputdir, 'unmasked', images[0]['file_name']))
    if len(mask) > 0:
        m_arr = imantics.Annotation.from_polygons(mask, image=imantics.Image(im_arr))
        m_arr = np.expand_dims(m_arr.array.astype(np.float16), 2) * mask_alpha
    else:
        m_arr = None

    # convert BoxMode, remove boxes overlapping with mask
    for im in images:
        annotations_filter = []
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
            if ann['bbox_mode'] != BoxMode.XYXY_ABS:
                raise 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])
            if not _check_overlap(m_arr, ann['bbox']):
                annotations_filter.append(ann)
        im['annotations'] = annotations_filter
    print('annotation: %d images %d bboxes' % (len(images), sum(map(lambda x: len(x['annotations']), images))))

    detections_matched = []
    for im in images:
        image_f = im['file_name']
        for i in range(0, len(detections)):
            if detections[i]['file_name'] == image_f:
                detections_matched.append(detections[i])
                break
    assert len(images) == len(detections), 'mismatch: not found in detection results'
    for im in detections_matched:
        annotations_filter = []
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
            if ann['bbox_mode'] != BoxMode.XYXY_ABS:
                raise 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])
            if not _check_overlap(m_arr, ann['bbox']):
                annotations_filter.append(ann)
        im['annotations'] = annotations_filter
    detections = detections_matched
    print('detections: %d images %d bboxes' % (len(detections), sum(map(lambda x: len(x['annotations']), detections))))
    if check_empty:
        _count = sum(map(lambda ann: len(ann['annotations']), detections))
        if _count < 1:
            print('no object detected, insert some random bboxes')
            for im in detections:
                x1, x2 = sorted((np.random.rand(2) * im['width'] * 0.8 + 2).tolist())
                y1, y2 = sorted((np.random.rand(2) * im['height'] * 0.8 + 2).tolist())
                im['annotations'].append({'bbox': list(map(float, [x1, y1, x2, y2])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0, 'score': 0.5})

    results = eval_AP(images, detections, return_thres=False)
    if not outputfile is None:
        fontsize, linewidth = int(im_arr.shape[0] * 0.012), int(im_arr.shape[0] / 400)
        font_label = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=fontsize)
        font_title = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=fontsize * 4)

        if outputfile[-4:] != '.mp4':
            outputfile = outputfile + '.mp4'
        writer = skvideo.io.FFmpegWriter(outputfile, inputdict={'-r': '1'}, outputdict={'-vcodec': 'libx265', '-r': '1', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
        for i in tqdm.tqdm(range(0, len(images)), ascii=True, desc='...' + outputfile[-40:]):
            im, det = images[i], detections[i]
            assert im['file_name'] == det['file_name']
            im_arr = skimage.io.imread(os.path.join(inputdir, 'unmasked', im['file_name']))
            if not m_arr is None:
                im_arr = ((1 - m_arr) * im_arr + m_arr * (np.array(mask_rgb) * 255).astype(np.uint8).reshape(1, 1, 3)).astype(np.uint8)
            im_arr_ann, im_arr_det = im_arr, im_arr.copy()

            im_arr_ann = Image.fromarray(im_arr_ann, 'RGB')
            draw = ImageDraw.Draw(im_arr_ann)
            for ann in im['annotations']:
                x1, y1, x2, y2 = ann['bbox']
                c_id = ann['category_id']
                draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[c_id], width=linewidth)
                draw.text((x1 + 3, y1 + 1), '%d %s' % (c_id, thing_classes[c_id]), fill=bbox_rgbs[c_id], font=font_label)
            draw.text((2, 2), 'Annotations', fill='#000000', stroke_width=3, font=font_title)
            draw.text((2, 2), 'Annotations', fill='#FFFFFF', stroke_width=1, font=font_title)
            im_arr_ann = np.array(im_arr_ann)

            im_arr_det = Image.fromarray(im_arr_det, 'RGB')
            draw = ImageDraw.Draw(im_arr_det)
            for ann in det['annotations']:
                x1, y1, x2, y2 = ann['bbox']
                c_id, s = ann['category_id'], ann['score']
                draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[c_id], width=linewidth)
                draw.text((x1 + 3, y1 + 1), '%d %s %.1f' % (c_id, thing_classes[c_id], s * 100), fill=bbox_rgbs[c_id], font=font_label)
            draw.text((2, 2), 'Detections', fill='#000000', stroke_width=3, font=font_title)
            draw.text((2, 2), 'Detections', fill='#FFFFFF', stroke_width=1, font=font_title)
            im_arr_det = np.array(im_arr_det)

            writer.writeFrame(np.concatenate([im_arr_ann, im_arr_det], axis=1))
        writer.close()
    if return_bboxes:
        return results, images, detections
    else:
        return results


def evaluate_cocovalid(cocodir, detections):
    class Dummy(object):
        pass
    from base_detector_train import get_coco_dicts
    args = Dummy()
    args.cocodir, args.smallscale = cocodir, False
    cocovalid_dict = get_coco_dicts(args, 'valid')
    for im in cocovalid_dict:
        im['file_name'] = os.path.basename(im['file_name'])
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
            if ann['bbox_mode'] != BoxMode.XYXY_ABS:
                raise 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])

    for i in range(0, len(detections)):
        detections[i]['file_name'] = os.path.basename(detections[i]['file_name'])
        assert detections[i]['file_name'] == cocovalid_dict[i]['file_name'], '%s %s' % (detections[i]['file_name'], cocovalid_dict[i]['file_name'])
        for ann in detections[i]['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
            if ann['bbox_mode'] != BoxMode.XYXY_ABS:
                raise 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])
    print('detections: %d images %d bboxes' % (len(detections), sum(map(lambda x: len(x['annotations']), detections))))
    return eval_AP(cocovalid_dict, detections)


if __name__ == '__main__':
    # with open(os.path.join(os.path.dirname(__file__), '..', 'images', 'annotated', '001', 'annotations.json'), 'r') as fp:
    #     detections = json.load(fp)
    # for im in detections:
    #     for ann in im['annotations']:
    #         ann['score'] = 0.9
    # # results = evaluate_masked('001', detections, outputfile='001.eval.mp4')
    # results = evaluate_masked('001', detections)
    # print(results)

    evaluate_cocovalid(os.path.join(os.path.dirname(__file__), '..', '..', 'MSCOCO2017'), None)
