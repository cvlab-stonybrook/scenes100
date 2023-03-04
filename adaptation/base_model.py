#!python3

import os
import sys
from detectron2 import model_zoo
from detectron2.config import get_cfg

sys.path.append(os.path.join(os.path.dirname(__file__)))
from constants import thing_classes


def get_cfg_base_model(m, ckpt=None):
    models = {
        'r50-fpn-3x': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'r101-fpn-3x': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    }
    assert m in models, 'model %s not recognized' % m

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(models[m]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    if not ckpt is None:
        assert os.access(ckpt, os.R_OK), '%s not readable' % ckpt
        cfg.MODEL.WEIGHTS = ckpt
        cfg.MODEL.WEIGHTS = os.path.normpath(cfg.MODEL.WEIGHTS)

    print('detectron2 model:', m)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    return cfg


if __name__ == '__main__':
    get_cfg_base_model('r101-fpn-3x')
    get_cfg_base_model('r101-fpn-1x')
