
python inference_server_simulate_dino.py \
    --train_whole 1 \
    --opt adapt \
    --config ../../configs/dino_5scale.yaml \
    --ckpt ../../models/dino_5scale_remap_orig.pth \
    --tag budget100 \
    --iters 10000 \
    --eval_interval 2000 \
    --save_interval 2000 \
    --image_batch_size 2 \
    --num_workers 4 \
    --outputdir ./dino_x2_split_b100_2stage \
    --lr 1e-4 \
    --refine_det_score_thres 0.3 \
    --budget 100 \
    # --interm \ # enable for intermediate layer split

    # --ckpt ../../models/dino_b1_x2_split_iters8kbase.pth \ # use this for 2-stage training

