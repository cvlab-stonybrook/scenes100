
python inference_server_simulate_dino.py \
    --train_whole 1 \
    --opt adapt \
    --ckpt ../../models/dino_5scale_remap.pth \
    --config ../../configs/dino_5scale.yaml \
    --tag budget10 \
    --iters 32000 \
    --eval_interval 2000 \
    --save_interval 2000 \
    --image_batch_size 2 \
    --num_workers 4 \
    --outputdir ./dino_x2_split_b10_1stage \
    --lr 1e-4 \
    --refine_det_score_thres 0.3 \
    --budget 10 \
    --mapper your_mapper/path/here.pth \
    # --interm \ # enable for intermediate layer split
    # --mapper ./dino_x2_split_b1/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.budget1.frombase.10means.fpn.p5.mapper.pth \
    # --mapper ./dino_x2_split_b1/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.budget1.10means.fpn.p5.mapper.pth \

    # --ckpt ../../models/dino_b1_x2_split_iters8kbase.pth \ # use this for 2-stage training