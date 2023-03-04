#!python3

import os
import sys
import json
from detectron2.structures import BoxMode

sys.path.append(os.path.join(os.path.dirname(__file__)))
from constants import thing_classes, thing_classes_coco
assert len(thing_classes_coco) == len(thing_classes)


def get_coco_dicts(split, use_background=False, segment=False):
    basedir = os.path.normpath( os.path.join(os.path.dirname(__file__), '..', 'mscoco'))
    if split == 'valid':
        annotations_json = os.path.join(basedir, 'instances_val2017.json')
    elif split == 'train':
        annotations_json = os.path.join(basedir, 'instances_train2017.json')
    else:
        raise NotImplementedError
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    category_id_remap = {}
    for cat in annotations['categories']:
        for i in range(0, len(thing_classes_coco)):
            if cat['name'] in thing_classes_coco[i]:
                category_id_remap[cat['id']] = i

    coco_dicts = {}
    images_dir = os.path.join(basedir, 'images', 'val2017' if split == 'valid' else 'train2017')
    background_dir = os.path.join(basedir, 'inpaint_mask', 'val2017' if split == 'valid' else 'train2017')
    for im in annotations['images']:
        coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
        if use_background:
            coco_dicts[im['id']]['file_name_background'] = os.path.join(background_dir, im['file_name'])
    for ann in annotations['annotations']:
        if not ann['category_id'] in category_id_remap:
            continue
        coco_dicts[ann['image_id']]['annotations'].append({'bbox': ann['bbox'], 'iscrowd': ann['iscrowd'], 'bbox_mode': BoxMode.XYWH_ABS, 'segmentation': ann['segmentation'] if segment else [], 'area': ann['area'], 'category_id': category_id_remap[ann['category_id']]})
    coco_dicts = list(coco_dicts.values())
    coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
    for i in range(0, len(coco_dicts)):
        coco_dicts[i]['image_id'] = i + 1
    count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('MSCOCO-2017 %s: %d images, %d bboxes' % (split, count_images, count_bboxes))
    return coco_dicts


if __name__ == '__main__':
    dst = get_coco_dicts('train')
    dst = get_coco_dicts('valid', use_background=True)
