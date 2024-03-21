#!/bin/bash

# This script is used to construct DIMTDA model and finetune it on the DoTA dataset.
base_dir=/path/to/DIMTDA

trans_model_dir=$base_dir/models/trans_model/checkpoint-xxxx
dit_model_dir=$base_dir/pretrained_models/dit-base
nougat_model_dir=$base_dir/pretrained_models/nougat-small
dimtda_model_dir=$base_dir/models/dimtda_model

image_processor_dir=$base_dir/utils/image_processor
zh_tokenizer_dir=$base_dir/utils/zh_tokenizer
qformer_config_dir=$base_dir/utils/blip2-opt-2.7b

image_dir=$base_dir/DoTA_dataset/imgs
zh_mmd_dir=$base_dir/DoTA_dataset/zh_mmd
split_json_file_path=$base_dir/DoTA_dataset/split_dataset.json

export CUDA_VISIBLE_DEVICES=0,1,2,3

python codes/finetune_dimtda.py \
    --trans_model_dir $trans_model_dir \
    --dit_model_dir $dit_model_dir \
    --nougat_model_dir $nougat_model_dir \
    --image_processor_dir $image_processor_dir \
    --zh_tokenizer_dir $zh_tokenizer_dir \
    --image_dir $image_dir \
    --zh_mmd_dir $zh_mmd_dir \
    --split_json_file_path $split_json_file_path \
    --output_dir $dimtda_model_dir \
    --qformer_config_dir $qformer_config_dir \
    --batch_size_per_gpu 4
