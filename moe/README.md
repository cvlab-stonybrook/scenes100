# Efficiency-Preserving Scene-Adaptive Object Detection

This repository is the official implementation of paper:

[Zekun Zhang](https://zvant.github.io/), [Vu Quang Truong](https://truong2710-cyber.github.io/), [Minh Hoai](https://www3.cs.stonybrook.edu/~minhhoai/), *Efficiency-Preserving Scene-Adaptive Object Detection*, BMVC 2024 (oral).

[[PDF](../media/efficienntSceneAdaptive-BMVC24.pdf)] [[Poster](../media/BMVC2024_poster.pdf)] [[Video Summary](../media/video_summary_moe.mp4)]

If you found our paper useful, please cite:
```bibtex
@inproceedings{ZhangetalBMVC24,
  author    = {Zekun Zhang and Vu Quang Truong and Minh Hoai},
  title     = {Efficiency-preserving Scene-adaptive Object Detection},
  booktitle = {Proceedings of British Machine Vision Conference (BMVC)},
  month     = {November},
  year      = {2024},
}
```

<p align="center">
  <img src="../media/moe.png" width="500">
</p>
<p  align="center">Network architecture of the MoE-enhanced model.</p>

## 1. Preparation
This implementation is based on the implementation our previous CVPR 2023 paper. See [README.md](../README.md) for the preparation of Scenes100, MSCOCO and the environment.

In addition, refer to [DINO](https://github.com/IDEA-Research/DINO) repo and use `pip install ultralytics` to install the requirements for DINO-5scale and YOLOv8s, respectively.

Finally, download the checkpoints in the following [Google Drive](https://drive.google.com/drive/folders/1ljqXfMDi-4QXJrYgEJ5ptLB_yNNCFPSK?usp=sharing) and put them in the folder `models`.

## 2. Run experiments

### 2.1 Faster-RCNN models

The code for Faster-RCNN based MoE models is in directory [script/fasterrcnn](script/fasterrcnn). ResNet-18 and ResNet-101 backbones are implemented. Please read arguments of the script `inference_server_simulate.py` for details. An example of 2-stage training workflow is shown below. Please note that the number of training iterations and batch size are reduced for quick running.

In the warmup stage, first train a $B$=1 model with
```console
python inference_server_simulate.py --train_whole 1 --opt adapt --model r18-fpn-3x --ckpt ../../models/mscoco2017_remap_r18-fpn-3x.pth --tag budget1 --budget 1 --iters 180 --eval_interval 100 --save_interval 150 --image_batch_size 2 --num_workers 2 --outputdir .
```
The trained model is saved to the checkpoint files `adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.pth` and `adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.mapper.pth`, which correspond to the warmed-up model. The next step is to get the gating rules. We need to use the warmed-up model to extract features from the images and apply clustering on the features. You need to have `scikit-learn` installed to perform the $K$-Means algorithm. For budget $B$=10, run
```console
python inference_server_simulate.py --model r18-fpn-3x --opt cluster --ckpts_dir . --ckpts_tag adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180 --budget 10
```
The clustering step requires much system memory and can be slow. After it finishes, pairs of `adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.10means.<FEATURE>.pth` and `adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.10means.<FEATURE>.mapper.pth` are created, where `<FEATURE>` is the feature used for clustering. Let us choose one to use
```console
mv adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.10means.fpn.p3.p4.pth r18.stage1.10means.pth
mv adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.10means.fpn.p3.p4.mapper.pth r18.stage1.10means.mapper.pth
```
Now we can start the second stage of training by running
```console
python inference_server_simulate.py --opt convert --ckpts_dir . --ckpts_tag adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180
python inference_server_simulate.py --train_whole 1 --opt adapt --model r18-fpn-3x --ckpt_teacher ../../models/mscoco2017_remap_r18-fpn-3x.pth --ckpt adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.budget1.iter.180.single.pth --tag r18.stage2.10means --resume_prefix r18.stage1.10means --budget 10 --iters 100 --eval_interval 200 --save_interval 200 --image_batch_size 2 --num_workers 2 --outputdir .
```
And the trained model is saved to `adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.r18.stage2.10means.iter.100.pth` and `adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.r18.stage2.10means.iter.100.mapper.pth`. To evaluate its performance, run
```console
python inference_server_simulate.py --model r18-fpn-3x --opt server --ckpts_dir . --ckpts_tag adapt_server_r18-fpn-3x_scenes100_pseudo_cocotrain.r18.stage2.10means.iter.100
```

### 2.2 YOLOv8 and DINO-5scale models

#### Training
To run the adaptation training of the MoE models, please use the `train_b1.sh`, `train_b10.sh` and `train_b100.sh` in `script/<ARCHITECTURE>`. You can modify the arguments to perform 1-stage and 2-stage training. The file `create_mapper.sh` is used to generate the mapper from a model checkpoint, including random and $B$-Means gating. Please check the arguments help information on how to use them.

Specifically:

 - Modify `--ckpt` to choose the starting checkpoint of the model. If you use 1-stage training, make sure `--ckpt` is the base model. Otherwise, `--ckpt` should be the warmed-up model.
 - Modify `--mapper` to choose the mapper for the model. You can leave it empty to use random gating or specify a `.pth` file for it for $B$-Means gating or any mapper you want. 

#### Evaluate Detection Performance
Use the `eval_<ARCHITECTURE>.py` file for evaluation. Please check the arguments help information on how to use it. You can use `--opt server` to get the AP score or `--opt tp` to measure throughput of the model. For example:
```console
# Measure throughput of DINO-5scale base model
python eval_dino.py --opt tp --ckpt ../../models/dino_5scale_remap_orig.pth --scale 1 --image_batch_size 4
# Get AP score of DINO-5scale base model
python eval_dino.py --opt server --ckpt ../../models/dino_5scale_remap_orig.pth --scale 1 --image_batch_size 4
```
 