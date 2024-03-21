import os
import json
import jieba
import re
import argparse
from sacrebleu import corpus_bleu, sentence_bleu
from zss import simple_distance, Node

def get_zh_text(zh_file_path):
    split_lines = []
    with open(zh_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(' '.join(jieba.cut(line.strip())) + ' \n\n')
    return ' '.join(split_lines)

def pre_clean(text):
    text = re.sub(r'<bos>|<eos>|<pad>|<unk>', '', text)
    text = re.sub(r'\s##(\S)', r'\1', text)
    text = re.sub(r'\\\s', r'\\', text)
    text = re.sub(r'\s\*\s\*\s', r'**', text)
    text = re.sub(r'{\s', r'{', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\\begin\s', r'\\begin', text)
    text = re.sub(r'\\end\s', r'\\end', text)
    text = re.sub(r'\\end{table}', r'\\end{table} \n\n', text)
    text = text.replace('\n', ' ')
    text = text.replace('*', ' ')
    text = text.replace('_', ' ')
    return text

def metric_post_process(text):
    text = pre_clean(text)
    text = text.replace('#', ' ')
    return text

def clean_table_math(text):
    text = re.sub(r'\\begin\{table\}.*?\\end\{table\}', ' ', text)
    text = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', ' ', text)
    text = re.sub(r'\\\(.*?\\\)', ' ', text)
    text = re.sub(r'\\\[.*?\\\]', ' ', text)
    return text

def get_tree(mmd_file_path):
    tree = (Node('ROOT').addkid(Node('TITLE')))
    with open(mmd_file_path, 'r', encoding='utf-8') as f:
        mmd_lines = f.readlines()
    lines = []
    for line in mmd_lines:
        line = pre_clean(line)
        if line.strip() != '':
            lines.append(line.strip())
    last_title = ''
    for line in lines:
        if line.startswith('#'):
            child = tree.get('ROOT')
            line = line.replace('#', '')
            child.addkid(Node(line))
            last_title = line
        else:
            if last_title == '':
                child = tree.get('TITLE')
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)
                child.addkid(Node(line))
    return tree

def STEDS(pred_tree, ref_tree):
    def my_distance(pred, ref):
        if len(pred.split()) == 0 or len(ref.split()) == 0:
            return 1
        else:
            return 0
    total_distance = simple_distance(pred_tree, ref_tree, label_dist=my_distance)
    num_of_nodes = max(len(list(pred_tree.iter())), len(list(ref_tree.iter())))
    return 1-total_distance/num_of_nodes

def evaluate(args):
    pred_dir = args.result_dir
    ref_dir = args.zh_mmd_dir

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    test_name_list = json_dict['test_name_list']

    pred_list = []
    ref_list = []
    pred_clean_math_table_list = []
    ref_clean_math_table_list = []
    STEDS_list = []
    from tqdm import tqdm
    for name in tqdm(test_name_list):
        pred_mmd_file_path = os.path.join(pred_dir, name + '.mmd')
        with open(pred_mmd_file_path, 'r') as f:
            lines = f.readlines()
        pred_mmd = ' '.join(lines)
        pred_mmd = metric_post_process(pred_mmd)
        pred_list.append(pred_mmd)
        pred_mmd = clean_table_math(pred_mmd)
        pred_clean_math_table_list.append(pred_mmd)
        
        ref_mmd_file_path = os.path.join(ref_dir, name + '.mmd')
        ref_mmd = get_zh_text(ref_mmd_file_path)
        ref_mmd = metric_post_process(ref_mmd)
        ref_list.append(ref_mmd)
        ref_mmd = clean_table_math(ref_mmd)
        ref_clean_math_table_list.append(ref_mmd)
        
        pref_tree = get_tree(pred_mmd_file_path)
        ref_tree = get_tree(ref_mmd_file_path)
        STEDS_list.append(STEDS(pref_tree, ref_tree))

    print('BLEU: %.2f BLEU-PT: %.2f STEDS: %.2f' % (corpus_bleu(pred_list, [ref_list]).score, corpus_bleu(pred_clean_math_table_list, [ref_clean_math_table_list]).score, sum(STEDS_list)/len(STEDS_list) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    
    args = parser.parse_args()
    
    evaluate(args)