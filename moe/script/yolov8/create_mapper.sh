#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov8.py \
        --opt cluster \
        --budget 10 \
        --model r101-fpn-3x \
        --ckpt ../../models/yolov8s_remap.pth \
        --ckpts_dir ./yolov8s_bs28_lr0.0001_teacherx2_conf0.4_b1 \
        --ckpts_tag adaptive_partial_server_yolov3_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.25999 \
        --config ../../configs/yolov8s.yaml \
        --image_batch_size 28 \
        --split_list 0 1 2 3 4 \
        --from_base 1 \

        # --random 1 \
        # --gen_seed 10 20 30 \
        # --outputdir ./random_mapping \