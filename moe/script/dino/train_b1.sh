python inference_server_simulate_dino.py \
    --train_whole 1 \
    --opt adapt \
    --ckpt ../../models/dino_5scale_remap_orig.pth \
    --config ../../configs/dino_5scale.yaml \
    --tag budget1 \
    --budget 1 \
    --iters 32000 \
    --eval_interval 2000 \
    --save_interval 2000 \
    --image_batch_size 2 \
    --num_workers 4 \
    --outputdir ./dino_x2_split_b1 \
    --lr 1e-4 \
    --refine_det_score_thres 0.3 \
    --id 001

