python inference_server_simulate_dino.py \
    --opt cluster \
    --ckpt ../../models/dino_5scale_remap_orig.pth \
    --ckpts_dir ./dino_x2_split_b1 \
    --ckpts_tag adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.budget1.frombase \
    --config ../../configs/dino_5scale.yaml \
    --image_batch_size 2 \
    --num_workers 4 \
    --budget 10
    # --ckpt ./dino_x2_split_b1/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.budget1.iter.7999.pth \
    # --ckpt ./dino_x1_base/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.vanilla.iter.23999.pth \

# python inference_server_simulate_dino.py \
#     --opt cluster \
#     --budget 10 \
#     --random 1 \
#     --gen_seed 140 150 160 170 180 190 200 \
#     --outputdir ./random_mapping \
