# Document Image Machine Translation with Dynamic Multi-pre-trained Models Assembling

This is the official repository for **DIMTDA** framework and **DoTA** dataset introduced by the following paper: [***Document Image Machine Translation with Dynamic Multi-pre-trained Models Assembling (NAACL 2024 Main)***](https://openreview.net/forum?id=XH2TgKlXWv)

## 📜 Abstract
Text image machine translation (TIMT) is a task that translates source texts embedded in the image to target translations.
The existing TIMT task mainly focuses on text-line-level images.
In this paper, we extend the current TIMT task and propose a novel task, **D**ocument **I**mage **M**achine **T**ranslation to **Markdown** (DIMT2Markdown), which aims to translate a source document image with long context and complex layout structure to markdownformatted target translation.
We also introduce a novel framework, **D**ocument **I**mage **M**achine **T**ranslation with **D**ynamic multi-pre-trained models **A**ssembling (DIMTDA).
A dynamic model assembler is used to integrate multiple pre-trained models to enhance the model’s understanding of layout and translation capabilities.
Moreover, we build a novel large-scale **Do**cument image machine **T**ranslation dataset of **A**rXiv articles in markdown format (DoTA), containing 126K image-translation pairs.
Extensive experiments demonstrate the feasibility of end-to-end translation of rich-text document images and the effectiveness of DIMTDA.

**The diagram of the proposed DIMTDA.**
![](images/fig_model.png)

**The output samples of DIMTDA.** (a) and (c) are the original document images. (b) and (d) are the output translated texts in markdown format after rendering.
![](images/fig_samples.png)

## 🗂️ DoTA dataset
In addition to the 126K samples mentioned in the paper, we provide all 139K samples that have not been filtered.
Each sample contains original English image, transcripted English mmd file and translated Chinese/French/German mmd file.
Samples used in the paper are listed in a json file.

The DoTA dataset can be downloaded from this [huggingface link](https://huggingface.co/datasets/liangyupu/DoTA_dataset).

## 🛠️ DIMTDA
### 1. Requirements
```bash
python==3.10.13
pytorch==1.13.1
transformers==4.33.2
sacrebleu==2.3.1
jieba==0.42.1
zss==1.2.0
```

### 2. Download pre-trained models
Download pre-trained DiT model from [microsoft/dit-base](https://huggingface.co/microsoft/dit-base).

Download pre-trained Nougat model from [facebook/nougat-small](https://huggingface.co/facebook/nougat-small).

The file directory structure is as follows:
```bash
DIMTDA
├── codes
├── DoTA_dataset
├── pretrained_models
└── utils
```

### 3. Pre-train a text translation model
```bash
bash pretrain_trans.sh
```

### 4. Finetune DIMTDA
```bash
bash finetune_dimtda.sh
```

### 5. Inference
Before running the script, you need to replace the `~/anaconda3/envs/your_env_name/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py` file with the `./utils/modeling_bert.py` file.
```bash
bash inference.sh
```

### 6. Evaluate
```bash
bash evaluate.sh
```

## 🖻 More samples
The output samples of DIMTDA. For each image pair, the left one is the input document image, and the right one is the output translations in markdown format after rendering.
![](images/fig_appendix.png)


## 🙏🏻 Acknowledgement
We thank @lukas-blecher and [facebookresearch/nougat](https://github.com/facebookresearch/nougat) project for providing dataset construction method and pre-trained model.
We also thank [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/dit) project for providing pre-trained model.

## ✍🏻 Citation
If you want to cite our paper, please use the following BibTex entries:
```BibTex
@inproceedings{liang-etal-2024-document,
    title = "Document Image Machine Translation with Dynamic Multi-pre-trained Models Assembling",
    author = "Liang, Yupu  and
      Zhang, Yaping  and
      Ma, Cong  and
      Zhang, Zhiyang  and
      Zhao, Yang  and
      Xiang, Lu  and
      Zong, Chengqing  and
      Zhou, Yu",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.392",
    pages = "7077--7088",
}
```

If you have any question, feel free to contact [liangyupu2021@ia.ac.cn](mailto:liangyupu2021@ia.ac.cn).
