#! /bin/sh

CUDA_VISIBLE_DEVICES='0' nohup python3 -u main.py --mode bi \
    --init_model_dir ./models/bert-base-uncased \
    --save_model_dir ./models/bi-bert-base-avg/ \
    --pool_type avg --temperature 0.5 \
    --batch_size 32 --max_length 128 \
    --train_path ./dataset/train.tsv  --valid_path ./dataset/dev.tsv  --test_path ./dataset/test.tsv
