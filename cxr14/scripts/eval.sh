python main.py \
    --run_name eval_acm \
    --path_to_images <PATH_TO_CXR14_DATA> \
    --project cxr14 \
    --orth_loss_lambda 0.1 \
    --gpu_ids 4,5,6,7 \
    --lr 0.04 \
    --batch_size 64 \
    --arch resnet50 \
    --input_size 448 \
    --random_crop True \
    --patience 4 \
    --module acm \
    --num_acm_groups 32 \
    --eval_only True \
    --run_path ./cLFoIvvZ


