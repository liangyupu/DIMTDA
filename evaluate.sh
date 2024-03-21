#!/bin/bash

# This script is used to calcaute BLEU, BLEU-PT and STEDS.
base_dir=/path/to/DIMTDA

split_json_file_path=$base_dir/DoTA_dataset/split_dataset.json
zh_mmd_dir=$base_dir/DoTA_dataset/zh_mmd
result_dir=$base_dir/results

python codes/evaluate.py \
    --split_json_file_path $split_json_file_path \
    --result_dir $result_dir \
    --zh_mmd_dir $zh_mmd_dir
