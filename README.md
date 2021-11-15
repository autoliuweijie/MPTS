# README

This is the dataset and code for the paper of "Semantic Matching from Different Perspectives"


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


## Supported models

| Model name            | URL |
|-----------------------|-----|
| bert-large-uncased    | https://huggingface.co/bert-large-uncased |
| bert-base-uncased     | https://huggingface.co/bert-base-uncased  |
| bert-medium-uncased   | https://huggingface.co/google/bert_uncased_L-8_H-512_A-8|
| bert-small-uncased    | https://huggingface.co/google/bert_uncased_L-4_H-512_A-8|
| bert-mini-uncased     | https://huggingface.co/google/bert_uncased_L-4_H-256_A-4|
| bert-tiny-uncased     | https://huggingface.co/google/bert_uncased_L-2_H-128_A-2|
| roberta-base          | https://huggingface.co/roberta-base      | 
| roberta-large         | https://huggingface.co/roberta-large     |
| sbert-base            | https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens |
| sbert-large           | https://huggingface.co/sentence-transformers/bert-large-nli-mean-tokens|
| simcse-bert-base      | https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased |
| simcse-bert-large     | https://huggingface.co/princeton-nlp/unsup-simcse-bert-large-uncased |
| simcse-roberta-base   | https://huggingface.co/princeton-nlp/unsup-simcse-roberta-base |
| simcse-roberta-large  | https://huggingface.co/gaotianyu1350/unsup-simcse-roberta-large|









