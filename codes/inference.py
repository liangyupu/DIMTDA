import os
import json
import torch
import jieba
import re
import argparse
from PIL import Image

def inference(args):
    MAX_LENGTH = args.max_length
    os.makedirs(args.result_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from transformers import AutoTokenizer, DonutProcessor, BeitImageProcessor
    dit_processor = BeitImageProcessor.from_pretrained(args.dit_model_dir)
    nougat_processor = DonutProcessor.from_pretrained(args.image_processor_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    test_name_list = json_dict['test_name_list']

    from transformers import EncoderDecoderModel, VisionEncoderDecoderModel, BeitModel, EncoderDecoderConfig
    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir)
    dit_model = BeitModel.from_pretrained(args.dit_model_dir)
    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir)
    
    from my_model import DIMTDAModel
    my_config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    model = DIMTDAModel(my_config, trans_model, dit_model, nougat_model, args.num_queries, args.qformer_config_dir)
    
    checkpoint_file_path = os.path.join(args.dimtda_model_dir, 'pytorch_model.bin')
    checkpoint = torch.load(checkpoint_file_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    from transformers import GenerationConfig
    generation_config = GenerationConfig(
        max_length=MAX_LENGTH,
        early_stopping=True,
        num_beams=args.num_beams,
        use_cache=True,
        length_penalty=1.0,
        bos_token_id=zh_tokenizer.bos_token_id,
        pad_token_id=zh_tokenizer.pad_token_id,
        eos_token_id=zh_tokenizer.eos_token_id,
    )

    from tqdm import tqdm
    for name in tqdm(test_name_list):
        image_file_path = os.path.join(args.image_dir, name+'.png')
        image = Image.open(image_file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        nougat_pixel_values = nougat_processor(image, return_tensors="pt").pixel_values.to(device)
        dit_pixel_values = dit_processor(image, return_tensors="pt").pixel_values.to(device)
        
        generation_ids = model.generate(
            nougat_pixel_values=nougat_pixel_values,
            dit_pixel_values=dit_pixel_values,
            generation_config=generation_config,
        )
        
        zh_text = zh_tokenizer.decode(generation_ids[0])
        
        result_file_path = os.path.join(args.result_dir, name+'.mmd')
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(zh_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_model_dir", type=str)
    parser.add_argument("--dit_model_dir", type=str)
    parser.add_argument("--nougat_model_dir", type=str)
    parser.add_argument("--dimtda_model_dir", type=str)
    parser.add_argument("--image_processor_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--qformer_config_dir", type=str)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--num_queries", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=4)

    args = parser.parse_args()
    
    inference(args)
