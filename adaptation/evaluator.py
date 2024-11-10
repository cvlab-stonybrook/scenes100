#!python3

import os
import sys
import json
import copy
import tempfile

import numpy as np
import imantics

import detectron2
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.join(os.path.dirname(__file__)))
from adaptation.constants import video_id_list, thing_classes
from adaptation.scenes100_dataset import get_manual_dicts


def eval_AP(images, detections):
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


def evaluate_masked(video_id, detections):
    assert video_id in video_id_list, 'unrecognizable ID: ' + video_id
    detections = copy.deepcopy(detections)
    images = get_manual_dicts(video_id)
    with open(os.path.join(os.path.dirname(__file__), '..', 'scenes100', 'masks.json'), 'r') as fp:
        mask = json.load(fp)
    mask = {m['video']: m['polygons'] for m in mask}[video_id]
    mask_alpha = 0.3

    im_arr = detectron2.data.detection_utils.read_image(images[0]['file_name'], format='BGR')
    if len(mask) > 0:
        mask = imantics.Annotation.from_polygons(mask, image=imantics.Image(im_arr))
        mask = np.expand_dims(mask.array.astype(np.float16), 2) * mask_alpha
    else:
        mask = None
    del im_arr

    # convert BoxMode, remove boxes overlapping with mask
    for im in detections:
        annotations_filter = []
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
            if ann['bbox_mode'] != BoxMode.XYXY_ABS:
                raise 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])
            if not _check_overlap(mask, ann['bbox']):
                annotations_filter.append(ann)
        im['annotations'] = annotations_filter

    assert len(images) == len(detections), 'mismatch'
    for im1, im2 in zip(images, detections):
        assert im1['file_name'] == im2['file_name'], 'mismatch: %s %s' % (im1['file_name'], im2['file_name'])
    return eval_AP(images, detections)


if __name__ == '__main__':
    pass
