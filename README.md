# README

This is the dataset and code for the paper of "Determine Similarity from Different Perspectives"


## Dataset

The MPTS dataset is in the ``./dataset/``


## Download models

Download the pre-trained model, take ``bert-base-uncased`` as an example.

```sh
$ cd models/
$ ./download_models bert-base-uncased
$ cd ../
```

## Training, Validation, and Evaluation

For ``bi-encoder`` mode, 
```sh
CUDA_VISIBLE_DEVICES='0' python3 -u main.py --mode bi \
    --init_model_dir ./models/bert-base-uncased \
    --save_model_dir ./models/bi-bert-base-avg/ \
    --pool_type avg --temperature 0.5 \
    --batch_size 32 --max_length 128 --num_epochs 10 \
    --train_path ./dataset/train.tsv  --valid_path ./dataset/dev.tsv  --test_path ./dataset/test.tsv
```

For ``cross-encoder`` mode,
```sh
CUDA_VISIBLE_DEVICES='0' python3 -u main.py --mode cross \
    --init_model_dir ./models/bert-base-uncased \
    --save_model_dir ./models/cross-bert-base-cls/ \
    --pool_type cls \
    --batch_size 32 --max_length 256 --num_epochs 10 \
    --train_path ./dataset/train.tsv  --valid_path ./dataset/dev.tsv  --test_path ./dataset/test.tsv
```





