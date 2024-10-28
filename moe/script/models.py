#!python3

import os
import json
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg

thing_classes = ['person', 'vehicle']


def get_cfg_base_model(desc, ckpt=None):
    with open(os.path.join(os.path.dirname(__file__), '..', 'models', 'models.json'), 'r') as fp:
        models = json.load(fp)
    models = {m['model']: m for m in models}
    assert desc in models, 'model %s not recognized' % desc
    m = models[desc]

    cfg = get_cfg()
    if m['type'] == 'official':
        cfg.merge_from_file(model_zoo.get_config_file(m['merge']))
    elif m['type'] == 'custom':
        cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', 'configs', m['merge']))
    else:
        raise Exception(m['type'])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    if not ckpt is None:
        assert os.access(ckpt, os.R_OK), '%s not readable' % ckpt
        cfg.MODEL.WEIGHTS = ckpt
    else:
        cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'models', m['ckpt'])
    cfg.MODEL.WEIGHTS = os.path.normpath(cfg.MODEL.WEIGHTS)

    print('detectron2 model:', desc)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    return cfg


if __name__ == '__main__':
    pass
