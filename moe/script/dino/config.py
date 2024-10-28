from detectron2.config import CfgNode as CN


def add_dino_config(cfg):
    """
    Add config for DINO.
    """
    cfg.MODEL.DINO = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("../../configs/dino_5scale.yaml")
