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


