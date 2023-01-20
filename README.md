# Overview
This is the code for paper [Effective Few-Shot Named Entity Linking by
Meta-Learning](https://arxiv.org/pdf/2207.05280.pdf).

## Preparation
MetaBLINK is built on the BLINK model, so firstly, you should clone and get familiar with [BLINK](https://github.com/facebookresearch/BLINK).
Please download the zero-shot entity linkint dataset and put the it under the `./data/zeshel` folder.
## Exact Matching
```
python generate_field_entities.py # for each domain (e.g., the Yugioh domain)
python pseudo_sample.py # for each domain (e.g., the Yugioh domain)
```
## Mention rewriting
Firstly, we collect training and validation data from the original dataset to train a T5 model for mention rewriting.
```
cd MR
python gen_data.py 
```
We use the huggingface's scripts for [unsupervised training](https://huggingface.co/docs/transformers/model_doc/t5) or [fine-tuning](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) T5.

Then you can rewrite the exact matching data using the fine-tuned T5.
And further filter out mistakes by running:
```
python filter.py
```
## Model Training
The model training procedure is almost the same as BLINK model. You can refer its repository for training details.

## Meta Learning
In our paper, we apply the meta-learning technique on BLINK. Please goto the `Meta-Learning` folder for more details.

