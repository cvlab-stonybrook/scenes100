#!/bin/bash
# 2-stage
CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov8.py \
    --train_whole 1 \
    --opt adapt \
    --ckpt ../../models/yolov8s_b1_x2_24k_base.pth \
    --config ../../configs/yolov8s.yaml \
    --tag seq.cluster.budget10 \
    --budget 10 \
    --iters 200 \
    --eval_interval 50 \
    --save_interval 100 \
    --image_batch_size 28 \
    --num_workers 4 \
    --outputdir ./test \
    --lr 1e-4 \
    --split_list 5 \
    --mapper yolov8s_bs28_lr0.0001_teacherx2_conf0.4_b1/adaptive_partial_server_yolov3_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.23999.10means.fpn.$2.new.mapper.pth
    # --mapper random_mapping/mapper_random_30_b10.pth \
# 11

# 1-stage
# CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov8.py \
#     --train_whole 1 \
#     --opt adapt \
#     --ckpt ../../models/yolov8s_remap.pth \
#     --config ../../configs/yolov8s.yaml \
#     --tag seq.cluster.budget10 \
#     --budget 10 \
#     --iters 34000 \
#     --eval_interval 2000 \
#     --save_interval 2000 \
#     --image_batch_size 28 \
#     --num_workers 4 \
#     --outputdir ./head/yolov8s_bs28_lr0.0001_teacherx2_conf0.4_b10_1stage_p4p5 \
#     --lr 1e-4 \
#     --split_list 15 16 \
#     --mapper yolov8s_bs28_lr0.0001_teacherx2_conf0.4_b1/10means.fpn.p4.p5.new.frombase.mapper.pth \
    # --id 001
    # --mapper random_mapping/mapper_random_30_b10.pth \