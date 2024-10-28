# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path
import detectron2
from detectron2.structures import Instances, Boxes, ImageList
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    Pose,
    RepC3,
    RepConv,
    ResNetLayer,
    RTDETRDecoder,
    Segment,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8OBBLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    make_divisible,
    model_info,
    scale_img,
    time_sync,
)
from ultralytics.utils.ops import non_max_suppression

try:
    import thop
except ImportError:
    thop = None

class MakeMoE(torch.nn.Module):
    def __init__(self, net, budget):
        super(MakeMoE, self).__init__()
        self.experts = torch.nn.ModuleList([deepcopy(net) for _ in range(budget)])
    def forward(self, x):
        assert len(x) == len(self.module_indices)
        out = [self.experts[m](x[i : i + 1, :]) for i, m in enumerate(self.module_indices)]
        self.module_indices = None
        return torch.cat(out, dim=0)

class MoEDetect(nn.Module):
    def __init__(self, head, budget):
        super(MoEDetect, self).__init__()
        self.net = head

        for attr in dir(head):
            if attr in ['f', 'i', 'stride', 'nc', 'nl', 'reg_max', 'no']:
                setattr(self, attr, getattr(head, attr))
        for i in range(len(self.net.cv3)):
            self.net.cv2[i][-1] = MakeMoE(self.net.cv2[i][-1], budget)
            self.net.cv3[i][-1] = MakeMoE(self.net.cv3[i][-1], budget)
        
    
    def forward(self, x):
        assert self.module_indices is not None, "did not assign."
        for i in range(len(self.net.cv3)):
            self.net.cv2[i][-1].module_indices = self.module_indices
            self.net.cv3[i][-1].module_indices = self.module_indices
        return self.net(x)


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward_image(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. "
            f"Reverting to single-scale inference instead."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        preds = self.forward_image(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose, OBB)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward_image(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward_image(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

        self.output_format = 'frcnn'

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)

    def reverse_yolo_transform(self, yolo_outputs, batched_inputs, output_format="frcnn"):
        reversed_outputs = []
        for i, output in enumerate(yolo_outputs):
            h_old, w_old = batched_inputs[i]['image'].shape[1:] # shape before padding
            if output_format == 'frcnn':                
                scores = output[:, 4].cuda()
                pred_classes = output[:, 5].cuda()
                reversed_boxes = output[:, :4]   

                boxes = Boxes(torch.Tensor(reversed_boxes).cuda())
                instances_dict = {'pred_boxes': boxes, 'scores': scores, 'pred_classes': pred_classes}
                instances = Instances(image_size=(h_old, w_old))
                for (k, v) in instances_dict.items():
                    instances.set(k, v)
                reversed_outputs.append(instances)
            elif output_format == "yolo":
                ratio = batched_inputs[i]['height'] / h_old
                output[:, :4] *= ratio
                reversed_outputs.append(output)
        return reversed_outputs

    @staticmethod
    def draw(image_tensor, detection_labels, format='yolo', input=True, ratio=None, file_name="test_preprocess.png"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        # Convert the PyTorch tensor to a NumPy array for visualization
        image_tensor = image_tensor.cpu()
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image_np)

        # Process the detection labels and draw bounding boxes
        if format == 'yolo':
            detection_labels = detection_labels.cpu()
            for detection in detection_labels:
                if input:
                    class_id, x, y, w, h = detection

                    # Convert normalized coordinates to absolute pixel coordinates
                    img_h, img_w = image_tensor.shape[1], image_tensor.shape[2]
                    x1 = (x - w/2) * img_w
                    y1 = (y - h/2) * img_h
                    x2 = (x + w/2) * img_w
                    y2 = (y + h/2) * img_h
                else:
                    assert ratio is not None, "ratio unavailable"
                    x1, y1, x2, y2, _, _ = detection 
                    x1 *= ratio
                    x2 *= ratio
                    y1 *= ratio
                    y2 *= ratio

                # Create a Rectangle patch
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')

                # Add the bounding box to the plot
                ax.add_patch(rect)
        elif format == 'frcnn':
            assert ratio is not None, "ratio unavailable"
            detection_labels = detection_labels.pred_boxes.tensor.cpu()
            for detection in detection_labels:
                x1, y1, x2, y2 = detection
                x1 *= ratio
                x2 *= ratio
                y1 *= ratio
                y2 *= ratio

                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.savefig(file_name)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        processed_batch = {}
        images = ImageList.from_tensors([im['image'].cuda()/255 for im in batched_inputs], size_divisibility=32).tensor  # padding to size dividable by 32
        images = images[:, (2, 1, 0), :, :] # BGR to RGB
        processed_batch['img'] = images
        for i, im in enumerate(batched_inputs):
            if self.training:
                
                assert 'instances' in batched_inputs[0], "no instances provided"
                # instances have to come from YOLO teacher or COCO 
                
                if isinstance(im['instances'], Instances):
                    # breakpoint()
                    # if the image is coco, the instances input would be in FRCNN format (i.e., Instances class)
                    instances = torch.zeros((im['instances'].gt_boxes.tensor.shape[0], 6))
                    instances[:, 2] = ((im['instances'].gt_boxes.tensor[..., 0] + im['instances'].gt_boxes.tensor[..., 2]) / 2) / images.shape[-1]
                    instances[:, 3] = ((im['instances'].gt_boxes.tensor[..., 1] + im['instances'].gt_boxes.tensor[..., 3]) / 2) / images.shape[-2]
                    instances[:, 4] = ((im['instances'].gt_boxes.tensor[..., 2] - im['instances'].gt_boxes.tensor[..., 0])) / images.shape[-1]
                    instances[:, 5] = ((im['instances'].gt_boxes.tensor[..., 3] - im['instances'].gt_boxes.tensor[..., 1])) / images.shape[-2]
                    
                    instances[:, 1] = im['instances'].gt_classes
                    instances[:, 0] = i
                else:
                    # output from YOLO teacher
                    # format: List[(xyxy, conf, cls)]. shape K x 6. Each element corresponding to an detection.
                    instances = im['instances'].clone() if isinstance(im['instances'], torch.Tensor) else np.copy(im['instances'])
                    
                    # XYXY abs to XYWH normalized
                    instances[:, 2] = ((im['instances'][..., 0] + im['instances'][..., 2]) / 2) / images.shape[-1]
                    instances[:, 3] = ((im['instances'][..., 1] + im['instances'][..., 3]) / 2) / images.shape[-2]
                    instances[:, 4] = ((im['instances'][..., 2] - im['instances'][..., 0])) / images.shape[-1]
                    instances[:, 5] = ((im['instances'][..., 3] - im['instances'][..., 1])) / images.shape[-2]
                    
                    instances[:, 1] = im['instances'][..., 5]
                    instances[:, 0] = i

                im['instances'] = instances

                # DetectionModel.draw(images[i], instances[:, 1:], file_name=f"input_afterprepocess_{i}.png")   

        if self.training:
            targets = torch.cat([im['instances'].cuda() for im in batched_inputs], dim=0).cuda()
            processed_batch['batch_idx'] = targets[:, 0].squeeze().cuda()
            processed_batch['cls'] = targets[:, 1].view(-1, 1).cuda()
            processed_batch['bboxes'] = targets[:, 2:].cuda()
            # breakpoint()
            return processed_batch
        else:
            return processed_batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], augment=False, profile=False, visualize=False):
        if not self.training: return self.inference(batched_inputs)
        batch = self.preprocess_image(batched_inputs)

        loss, loss_components = self.loss(batch)
        return loss, loss_components

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], augment=False, profile=False, visualize=False):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        x = self.preprocess_image(batched_inputs)

        yolo_outputs = self.predict(x['img'])  # single-scale inference, train

        conf_thres = 0.4 # teacher threshold = 0.4
        outputs = non_max_suppression(yolo_outputs[0], conf_thres=conf_thres, iou_thres=0.45)
        if self.output_format == 'yolo':            
            reversed_outputs = self.reverse_yolo_transform(outputs, batched_inputs, output_format='yolo')
            # for i in range(x.shape[0]):
            #     DetectionModel.draw(x[i], reversed_outputs[i], input=False, ratio=batched_inputs[i]['image'].shape[-2] / batched_inputs[i]['height'], file_name=f"output_teacher_{i}.png")
            # breakpoint()
            return reversed_outputs
        reversed_outputs = self.reverse_yolo_transform(outputs, batched_inputs)
        results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(reversed_outputs, batched_inputs, [im['image'].shape[1:] for im in batched_inputs])
        return results

Model = DetectionModel

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
        ):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose, OBB):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ""


def load_yolov8(config, weight=None):
    # model = parse_model(yaml_model_load(config), 3)[0]
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = DetectionModel(config).to(device)
    if weight is not None:
        model.load_state_dict(torch.load(weight))
    
    class Args:
        def __init__(self, input_dict):
            for key, value in input_dict.items():
                setattr(self, key, value)
    model.args = Args({**DEFAULT_CFG_DICT})
    return model

