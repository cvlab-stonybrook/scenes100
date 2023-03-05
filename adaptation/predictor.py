#!python3

import os
import sys
import copy
import time
import types
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

sys.path.append(os.path.join(os.path.dirname(__file__)))
from fusion_models import GeneralizedRCNNEarlyFusion, GeneralizedRCNNMidFusion, GeneralizedRCNNLateFusion


class AdaptationPredictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        if cfg.FUSION == 'vanilla':
            pass
        elif cfg.FUSION == 'earlyfusion':
            self.model = GeneralizedRCNNEarlyFusion.create_from_sup(self.model)
        elif cfg.FUSION == 'midfusion':
            self.model = GeneralizedRCNNMidFusion.create_from_sup(self.model, None)
        elif cfg.FUSION == 'latefusion':
            self.model = GeneralizedRCNNLateFusion.create_from_sup(self.model, None)
        else:
            raise NotImplementedError
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, image, image_diff=None):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            assert self.input_format == 'BGR'
            height, width = image.shape[:2]
            tf = self.aug.get_transform(image)
            image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
            if image_diff is None:
                inputs = {'image': image, 'height': height, 'width': width}
                return self.model.inference([inputs])
            else:
                image_diff = torch.as_tensor(tf.apply_image(image_diff).astype('float32').transpose(2, 0, 1))
                inputs = {'image': torch.cat([image, image_diff], dim=0), 'height': height, 'width': width}
                return self.model.inference([inputs])


if __name__ == '__main__':
    pass
