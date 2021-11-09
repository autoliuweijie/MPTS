#! /bin/sh


CUDA_VISIBLE_DEVICES='0' python3 -u main.py --mode cross \
    --init_model_dir ./models/bert-base-uncased \
    --save_model_dir ./models/cross-bert-base-cls/ \
    --pool_type cls \
    --batch_size 32 --max_length 256 --num_epochs 10 \
    --train_path ./dataset/train.tsv  --valid_path ./dataset/dev.tsv  --test_path ./dataset/test.tsvt
