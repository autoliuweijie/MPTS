#!/bin/bash


model_name=$1

if [ "${model_name}" = "bert-base-uncased" ]; then

    model_url="https://huggingface.co/bert-base-uncased"
    model_file_name="bert-base-uncased"
    model_name="bert-base-uncased"

elif [ "${model_name}" = "bert-medium-uncased" ]; then

    model_url="https://huggingface.co/google/bert_uncased_L-8_H-512_A-8"
    model_file_name="bert_uncased_L-8_H-512_A-8"
    model_name="bert-medium-uncased"

elif [ "${model_name}" = "bert-small-uncased" ]; then

    model_url="https://huggingface.co/google/bert_uncased_L-4_H-512_A-8"
    model_file_name="bert_uncased_L-4_H-512_A-8"
    model_name="bert-small-uncased"

elif [ "${model_name}" = "bert-mini-uncased" ]; then

    model_url="https://huggingface.co/google/bert_uncased_L-4_H-256_A-4"
    model_file_name="bert_uncased_L-4_H-256_A-4"
    model_name="bert-mini-uncased"

elif [ "${model_name}" = "bert-tiny-uncased" ]; then

    model_url="https://huggingface.co/google/bert_uncased_L-2_H-128_A-2"
    model_file_name="bert_uncased_L-2_H-128_A-2/"
    model_name="bert-tiny-uncased"

elif [ "${model_name}" = "bert-large-uncased" ]; then

    model_url="https://huggingface.co/bert-large-uncased"
    model_file_name="bert-large-uncased"
    model_name="bert-large-uncased"

elif [ "${model_name}" = "roberta-base" ]; then

    model_url="https://huggingface.co/roberta-base"
    model_file_name="roberta-base"
    model_name="roberta-base"

elif [ "${model_name}" = "roberta-large" ]; then

    model_url="https://huggingface.co/roberta-large"
    model_file_name="roberta-large"
    model_name="roberta-large"

elif [ "${model_name}" = "sbert-base" ]; then

    model_url="https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens"
    model_file_name="bert-base-nli-mean-tokens"
    model_name="sbert-base"

elif [ "${model_name}" = "sbert-large" ]; then

    model_url="https://huggingface.co/sentence-transformers/bert-large-nli-mean-tokens"
    model_file_name="bert-large-nli-mean-tokens"
    model_name="sbert-large"

elif [ "${model_name}" = "simcse-bert-base" ]; then

    model_url="https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased"
    model_file_name="unsup-simcse-bert-base-uncased"
    model_name="simcse-bert-base"

elif [ "${model_name}" = "simcse-bert-large" ]; then

    model_url="https://huggingface.co/princeton-nlp/unsup-simcse-bert-large-uncased"
    model_file_name="unsup-simcse-bert-large-uncased"
    model_name="simcse-bert-large"

elif [ "${model_name}" = "simcse-roberta-base" ]; then

    model_url="https://huggingface.co/princeton-nlp/unsup-simcse-roberta-base"
    model_file_name="unsup-simcse-roberta-base"
    model_name="simcse-roberta-base"

elif [ "${model_name}" = "simcse-roberta-large" ]; then

    model_url="https://huggingface.co/gaotianyu1350/unsup-simcse-roberta-large"
    model_file_name="unsup-simcse-roberta-large"
    model_name="simcse-roberta-large"

else
    echo "Unknown model: ${model_name}"
fi

if [ ! -r "$model_name" ]; then
    echo "Download ${model_name} from ${model_url}"
    git lfs clone ${model_url}
    mv ${model_file_name} ${model_name}
else
    echo "${model_name} already exists."
fi


echo "Finish."


