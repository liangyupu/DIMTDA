import os
import json
import torch
import jieba
import re
import argparse 

def train(args):
    MAX_LENGTH = args.max_length
    
    from transformers import AutoTokenizer, DonutProcessor, BeitImageProcessor
    dit_processor = BeitImageProcessor.from_pretrained(args.dit_model_dir)
    nougat_processor = DonutProcessor.from_pretrained(args.image_processor_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict['train_name_list']
    valid_name_list = json_dict['valid_name_list']

    from my_dataset import DoTADataset
    valid_dataset = DoTADataset(dit_processor, nougat_processor, zh_tokenizer, args.image_dir, args.zh_mmd_dir, valid_name_list, MAX_LENGTH)
    train_dataset = DoTADataset(dit_processor, nougat_processor, zh_tokenizer, args.image_dir, args.zh_mmd_dir, train_name_list, MAX_LENGTH)

    from transformers import EncoderDecoderModel, VisionEncoderDecoderModel, BeitModel, EncoderDecoderConfig
    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir)
    dit_model = BeitModel.from_pretrained(args.dit_model_dir)
    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir)
    
    from my_model import DIMTDAModel
    my_config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    model = DIMTDAModel(my_config, trans_model, dit_model, nougat_model, args.num_queries, args.qformer_config_dir)

    num_gpu = torch.cuda.device_count()
    gradient_accumulation_steps = args.batch_size // (num_gpu * args.batch_size_per_gpu)
    
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_strategy='steps',
        logging_steps=1,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(training_args)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_model_dir", type=str)
    parser.add_argument("--dit_model_dir", type=str)
    parser.add_argument("--nougat_model_dir", type=str)
    parser.add_argument("--image_processor_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--qformer_config_dir", type=str)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=1024)
    
    args = parser.parse_args()
    
    train(args)