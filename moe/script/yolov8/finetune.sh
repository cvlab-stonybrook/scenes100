#!/bin/bash

ids=('075' '076' '077' '080' '085' '086' '087' '088' '090' '091' '092' '093' '094' '095' '098' '099' '105' '108' '110' '112' '114' '115' '116' '117' '118' '125' '127' '128' '129' '130' '131' '132' '135' '136' '141' '146' '148' '149' '150' '152' '154' '156' '158' '159' '160' '161' '164' '167' '169' '170' '171' '172' '175' '178' '179')

for id in "${ids[@]}"; do
    echo "Finetune video $id"
    CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov8.py \
            --id $id \
            --train_whole 1 \
            --opt adapt \
            --model r101-fpn-3x \
            --ckpt ../../models/yolov8s_b1_x2_24k_base.pth \
            --config ../../configs/yolov8s.yaml  \
            --tag seq.cluster.budget1 \
            --budget 1 \
            --iters 100 \
            --eval_interval 100 \
            --save_interval 100 \
            --image_batch_size 28 \
            --num_workers 4 \
            --outputdir ./finetune_bs28_lr0.0001_teacherx2_conf0.4_continue24k_equal_iters/$id/ \
            --lr 1e-4 \
            --split_list 0 1 2 3 4 -1
done


# python finetune_oracle.py \
#         --opt adapt \
#         --train_r 0.25 \
#         --model r101-fpn-3x \
#         --ckpt ../../models/yolov8s_remap.pth \
#         --config ../../configs/yolov8s.yaml \
#         --iters 10000 \
#         --eval_interval 2000 \
#         --save_interval 2000 \
#         --image_batch_size 28 \
#         --num_workers 2 \
#         --outputdir ./finetune_oracle/0.25train/ \
#         --lr 1e-4 \
#         --budget 1