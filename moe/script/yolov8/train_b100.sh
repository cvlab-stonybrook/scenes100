#!/bin/bash
# 2 phases
# CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov8.py \
#     --train_whole 1 \
#     --opt adapt \
#     --ckpt ../../models/yolov8s_b1_x2_24k_base.pth \
#     --config ../../configs/yolov8s.yaml \
#     --tag seq.cluster.budget100 \
#     --budget 100 \
#     --iters 10000 \
#     --eval_interval 10000 \
#     --save_interval 10000 \
#     --image_batch_size 28 \
#     --num_workers 4 \
#     --outputdir ./head/yolov8s_bs28_lr0.0001_teacherx2_conf0.4_b100_continue24k_interm_12 \
#     --lr 1e-4 \
#     --split_list 12 \

# 1 phase
CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov8.py \
    --train_whole 1 \
    --opt adapt \
    --ckpt ../../models/yolov8s_remap.pth \
    --config ../../configs/yolov8s.yaml \
    --tag seq.cluster.budget100 \
    --budget 100 \
    --iters 50000 \
    --eval_interval 2000 \
    --save_interval 2000 \
    --image_batch_size 28 \
    --num_workers 4 \
    --outputdir ./head/yolov8s_bs28_lr0.0001_teacherx2_conf0.4_b100_1stage\
    --lr 1e-4 \
    --split_list 0 1 2 3 4 -1 \