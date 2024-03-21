import os
import json
import torch
import jieba
import re
import argparse 

def train(args):

    MAX_LENGTH = args.max_length
    
    from transformers import AutoTokenizer
    en_tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict['train_name_list']
    valid_name_list = json_dict['valid_name_list']

    from my_dataset import DoTADatasetTrans
    valid_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, valid_name_list, MAX_LENGTH)
    train_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, train_name_list, MAX_LENGTH)

    from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig

    encoder_config = BertConfig()
    decoder_config = BertConfig()
    encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    
    # Modify config according to transformer-base
    encoder_decoder_config.encoder.bos_token_id = en_tokenizer.bos_token_id
    encoder_decoder_config.encoder.eos_token_id = en_tokenizer.eos_token_id
    encoder_decoder_config.encoder.hidden_size = 512
    encoder_decoder_config.encoder.intermediate_size = 2048
    encoder_decoder_config.encoder.max_length = MAX_LENGTH
    encoder_decoder_config.encoder.max_position_embeddings = MAX_LENGTH
    encoder_decoder_config.encoder.num_attention_heads = 8
    encoder_decoder_config.encoder.num_hidden_layers = 6
    encoder_decoder_config.encoder.pad_token_id = en_tokenizer.pad_token_id
    encoder_decoder_config.encoder.type_vocab_size = 1
    encoder_decoder_config.encoder.vocab_size = len(en_tokenizer)

    encoder_decoder_config.decoder.bos_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.decoder.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.decoder.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.decoder.hidden_size = 512
    encoder_decoder_config.decoder.intermediate_size = 2048
    encoder_decoder_config.decoder.max_length = MAX_LENGTH
    encoder_decoder_config.decoder.max_position_embeddings = MAX_LENGTH
    encoder_decoder_config.decoder.num_attention_heads = 8
    encoder_decoder_config.decoder.num_hidden_layers = 6
    encoder_decoder_config.decoder.pad_token_id = zh_tokenizer.pad_token_id
    encoder_decoder_config.decoder.type_vocab_size = 1
    encoder_decoder_config.decoder.vocab_size = len(zh_tokenizer)

    encoder_decoder_config.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.pad_token_id = zh_tokenizer.pad_token_id
    encoder_decoder_config.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.max_length = MAX_LENGTH
    encoder_decoder_config.early_stopping = True
    encoder_decoder_config.no_repeat_ngram_size = 3
    encoder_decoder_config.length_penalty = 1.0
    encoder_decoder_config.num_beams = 4
    encoder_decoder_config.vocab_size = len(zh_tokenizer)

    model = EncoderDecoderModel(config=encoder_decoder_config)

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
        dataloader_num_workers=args.dataloader_num_workers
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
    parser.add_argument("--en_tokenizer_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--en_mmd_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    
    args = parser.parse_args()
    
    train(args)