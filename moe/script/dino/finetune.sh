#!/bin/bash

# ids=('001' '003' '005' '006' '007' '008' '009' '011' '012' '013' '014' '015' '016' '017' '019' '020' '023' '025' '027' '034' '036' '039' '040' '043' '044' '046' '048' '049' '050' '051' '053' '054' '055' '056' '058' '059' '060' '066' '067' '068' '069' '070' '071' '073' '074' '075' '076' '077' '080' '085' '086' '087' '088' '090' '091' '092' '093' '094' '095' '098' '099' '105' '108' '110' '112' '114' '115' '116' '117' '118' '125' '127' '128' '129' '130' '131' '132' '135' '136' '141' '146' '148' '149' '150' '152' '154' '156' '158' '159' '160' '161' '164' '167' '169' '170' '171' '172' '175' '178' '179')

# for id in "${ids[@]}"; do
#     echo "Finetune video $id"
#     CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_dino.py \
#             --id $id \
#             --train_whole 1 \
#             --opt adapt \
#             --model r101-fpn-3x \
#             --ckpt ../../models/dino_b1_x2_split_iters8kbase.pth \
#             --config ../../configs/dino_5scale.yaml \
#             --tag seq.cluster.budget1 \
#             --budget 1 \
#             --iters 20 \
#             --eval_interval 20 \
#             --save_interval 20 \
#             --image_batch_size 2 \
#             --num_workers 4 \
#             --outputdir ./finetune_bs2_lr0.0001_teacherx2split_conf0.3_continue8k_equal_iters_20/$id/ \
#             --lr 1e-4 \
#             --refine_det_score_thres 0.3
# done


python finetune_oracle.py \
        --opt adapt \
        --train_r 0.33 \
        --model r101-fpn-3x \
        --ckpt ../../models/dino_5scale_remap_orig.pth \
        --iters 30000 \
        --eval_interval 2000 \
        --save_interval 2000 \
        --image_batch_size 2 \
        --num_workers 2 \
        --outputdir ./finetune_oracle/0.33train_2/ \
        --lr 1e-4 \
        --budget 1
