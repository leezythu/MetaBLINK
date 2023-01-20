# Overview
This is the code for paper [1]
# Preparation
MetaBLINK is built on the BLINK model, so firstly, you should clone and get familiar with the original repository[2].
Please download the zero-shot entity linkint dataset and put the it under the `./data` folder.
# Exact Matching
```
python generate_field_entities.py # for each domain (e.g., the Yugioh domain)
python pseudo_sample.py # for each domain (e.g., the Yugioh domain)
```
# Mention rewriting
Firstly, we collect training and validation data from the original dataset to train a T5 model for mention rewriting.
```
cd MR
python gen_data.py 
```
We use the huggingface's scripts to [unsupervised training]{https://huggingface.co/docs/transformers/model_doc/t5} or [fine-tune]{https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization} T5.

Then you can rewrite the exact matching data using fine-tuned T5.
And further filter out mistakes by running:
`python filter.py`
# Model Training
The model training procedure is almost the same as BLINK model. You can refer its repository for more details.
# Meta Learning
Please see the `Meta-Learning` folder for more details.