#!/bin/bash

python inference_server_simulate_yolov8.py \
    --train_whole 1 \
    --opt adapt \
    --ckpt ../../models/yolov8s_remap.pth \
    --config ../../configs/yolov8s.yaml \
    --tag seq.cluster.budget1 \
    --budget 1 \
    --iters 40000 \
    --eval_interval 500 \
    --save_interval 2000 \
    --image_batch_size 28 \
    --num_workers 4 \
    --outputdir ./yolov8s_bs28_lr0.0001_teacherx2_conf0.4_b1 \
    --lr 1e-4 \
    --split_list 0 1 2 3 4 -1 \
    --id 001
