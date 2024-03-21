import os
import json
import torch
import jieba
import re
from PIL import Image

def get_en_text(en_file_path):
    split_lines = []
    with open(en_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(line.strip() + ' \n\n')
    return ' '.join(split_lines)

def get_zh_text(zh_file_path):
    split_lines = []
    with open(zh_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(' '.join(jieba.cut(line.strip())) + ' \n\n')
    return ' '.join(split_lines)

from torch.utils.data import Dataset
class DoTADatasetTrans(Dataset):
    def __init__(self, en_tokenizer, zh_tokenizer, en_txt_dir, zh_txt_dir, name_list, max_length):
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.en_txt_dir = en_txt_dir
        self.zh_txt_dir = zh_txt_dir
        self.name_list = name_list
        self.max_length = max_length
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        encoding = {}
        en_file_path = os.path.join(self.en_txt_dir, self.name_list[index]+'.mmd')
        en_text = get_en_text(en_file_path)
        tokenizer_outputs = self.en_tokenizer(en_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding['input_ids'] = tokenizer_outputs['input_ids'][0]
        encoding['attention_mask'] = tokenizer_outputs['attention_mask'][0]
        
        zh_file_path = os.path.join(self.zh_txt_dir, self.name_list[index]+'.mmd')
        zh_text = get_zh_text(zh_file_path)
        tokenizer_outputs = self.zh_tokenizer(zh_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding['decoder_input_ids'] = tokenizer_outputs['input_ids'][0]
        encoding['decoder_attention_mask'] = tokenizer_outputs['attention_mask'][0]
        input_ids = tokenizer_outputs['input_ids'][0].tolist()
        labels = input_ids[1:] + [-100]*(self.max_length-len(input_ids)+1)
        encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return encoding

class DoTADataset(Dataset):
    def __init__(self, dit_processor, nougat_processor, tokenizer, image_dir, text_dir, name_list, max_length):
        self.dit_processor = dit_processor
        self.nougat_processor = nougat_processor
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.name_list = name_list
        self.max_length = max_length
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        encoding = {}
        image_file_path = os.path.join(self.image_dir, self.name_list[index]+'.png')
        image = Image.open(image_file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        dit_pixel_values = self.dit_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        encoding['dit_pixel_values'] = dit_pixel_values
        nougat_pixel_values = self.nougat_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        encoding['nougat_pixel_values'] = nougat_pixel_values
        
        text_file_path = os.path.join(self.text_dir, self.name_list[index]+'.mmd')
        zh_text = get_zh_text(text_file_path)
        
        tokenizer_outputs = self.tokenizer(zh_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding['decoder_input_ids'] = tokenizer_outputs['input_ids'][0]
        encoding['decoder_attention_mask'] = tokenizer_outputs['attention_mask'][0]
        input_ids = tokenizer_outputs['input_ids'][0].tolist()
        labels = input_ids[1:] + [-100]*(self.max_length-len(input_ids)+1)
        encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return encoding
