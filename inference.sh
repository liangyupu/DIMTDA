#!/bin/bash

# This script is used to do inference and generate the Chinese texts for the input images.
base_dir=/path/to/DIMTDA

trans_model_dir=$base_dir/models/trans_model/checkpoint-xxxx
dit_model_dir=$base_dir/pretrained_models/dit-base
nougat_model_dir=$base_dir/pretrained_models/nougat-small
dimtda_model_dir=$base_dir/models/dimtda_model/checkpoint-xxxx

image_processor_dir=$base_dir/utils/image_processor
zh_tokenizer_dir=$base_dir/utils/zh_tokenizer
qformer_config_dir=$base_dir/utils/blip2-opt-2.7b

image_dir=$base_dir/DoTA_dataset/imgs
split_json_file_path=$base_dir/DoTA_dataset/split_dataset.json

result_dir=$base_dir/results

export CUDA_VISIBLE_DEVICES=0,1,2,3

python codes/inference.py \
    --trans_model_dir $trans_model_dir \
    --dit_model_dir $dit_model_dir \
    --nougat_model_dir $nougat_model_dir \
    --dimtda_model_dir $dimtda_model_dir \
    --image_processor_dir $image_processor_dir \
    --zh_tokenizer_dir $zh_tokenizer_dir \
    --image_dir $image_dir \
    --split_json_file_path $split_json_file_path \
    --result_dir $result_dir \
    --qformer_config_dir $qformer_config_dir
