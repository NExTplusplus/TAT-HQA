Learning to Imagine: Integrating Counterfactual Thinking in Neural Discrete Reasoning
====================

This repositary contains the **TAT-HQA** dataset and the code for ACL 2022 paper, view [PDF](https://aclanthology.org/2022.acl-long.5.pdf).

You can download our [TAT-HQA dataset](https://github.com/NExTplusplus/TAT-HQA/tree/master/dataset_raw).

# Dataset 

The hypothetical questions in TAT-HQA are created from the factual questions of [TAT-QA](https://github.com/NExTplusplus/TAT-QA). 
The mixed TAT-QA and TAT-HQA data is stored in [dataset_raw](https://github.com/NExTplusplus/TAT-HQA/tree/master/dataset_raw), where the corresponding factual question of each hypothetcial question is specified. The data files are organized by tables and a list of following passages, with a list of questions under each table. Refer to the website for detailed description on the data format. 

The questions contain the following keys, 
- `uid`: the unique question id.
- `order`: the order in the question list under the table and passages. 
- `question`: a string
- `answer`: a list of strings. The model is expected to generate all the answers. 
- 'derivation': for arithmetic questions, an equation of the answer calculation process. 
- `answer_type`: 4 types, span, multi-span, arithmetic or count. 
- `answer_from`: 3 types, table, text or table-text. 
- `rel_paragraph`: the order(s) of the relevant passage(s).
- `req_comparison`: True or False.
- `scale`: the model is also expected to predict a correct scale ('', thousand, million, billion, or percent) for each question. If the scale prediction is incorrect, the answer is evaluated as incorrect. 
- `rel_question (for TAT-HQA)`: the order of the corresponding factual question. Usually, it is the previous question. 
- `scale`: the model is also expected to predict a correct scale ('', thousand, million, billion, or percent) for each question. If the scale prediction is incorrect, the answer is incorrect. 

For our implementation of the paper method, we pre-process dataset_raw to generate some extra_fields. The `facts` and `mapping` are generated in the same way as TAT-QA (used for training the TagOP baseline). Apart from these, we extract the assumption substring from the hypothetical question (`question_if_part`), and we heuristically generate the `if_op`(SWAP, ADD, MINUS, DOUBLE, INCREASE_PERC, etc.) and `if_tagging` (the operand of if_op) for the Learning-to-imagine module. The preocessed data is stored in [dataset_extra_field] and splitted by TAT-QA and TAT-HQA, saved in `orig` and `counter`. 


## Requirements
The paper method is built upon `TagOp` ([Paper](https://aclanthology.org/2021.acl-long.254.pdf)) and further fine-tune on TAT-HQA from the TagOp checkpoint. Please follow the process in [TAT-QA](https://github.com/NExTplusplus/TAT-QA) to create the conda environment and download roberta.large. 
Our cuda version is 10.2, cuda driver version 440.33.01. We use one 32GB GPU. 

## Training & Testing

### Preprocess Dataset

Process the training/validation data for both TAT-QA and TAT-HQA. 

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_extra_field/[counter/orig] --output_dir tag_op/data/[counter/orig] --encoder roberta --mode [train/dev]
```

### Training

First, we obtain a checkpoint for TAT-QA by running the following command. Set --roberta_model as the path to the roberta model. 

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir tag_op/data/orig --save_dir tag_op/model_orig --batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4  --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 100 --eps 1e-6  --encoder roberta --test_data_dir tag_op/data/orig/ --roberta_model roberta.large --cross_attn_layer 0 --do_finetune 0
```

To evaluate the performance on TAT-QA, run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/orig --test_data_dir tag_op/data/orig --save_dir tag_op/model_orig --eval_batch_size 8 --model_path tag_op/model_orig --encoder roberta --roberta_model roberta.large
```
The predicted answer file is saved at `tag_op/model_orig/answer_dev.json`. The result will be on the validation set. 

Fine-tune TAT-HQA on with the L2I module by setting --do_finetune, --model_finetune_from tag_op/model_orig and --cross_attn_layer 3

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir tag_op/data/counter --save_dir tag_op/model_counter_ft_from_orig --batch_size 32 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-5  --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-6 --bert_weight_decay 0.01 --log_per_updates 100 --eps 1e-6  --encoder roberta --test_data_dir tag_op/data/counter/ --roberta_model roberta.large --cross_attn_layer 3 --do_finetune 1 --model_finetune_from tag_op/model_orig
```

### Testing
To test the performance on TAT-HQA, run the following command and obtain the prediction file at `tag_op/model_counter_ft_from_orig/answer_dev.json`. 
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/counter --test_data_dir tag_op/data/counter --save_dir tag_op/model_counter_ft_from_orig --eval_batch_size 8 --model_path tag_op/model_counter_ft_from_orig --encoder roberta --roberta_model roberta.large
```

### Evaluation
To use the evaluation script `evaluate.py`, try running

```bash
python evaluate.py dataset_extra_field/tatqa_and_hqa_field_dev.json tag_op/model_counter_ft_from_orig/answer_dev.json 0
```

## Citation 
```bash
@inproceedings{li2022learning,
  title={Learning to Imagine: Integrating Counterfactual Thinking in Neural Discrete Reasoning},
  author={Li, Moxin and Feng, Fuli and Zhang, Hanwang and He, Xiangnan and Zhu, Fengbin and Chua, Tat-Seng},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={57--69},
  year={2022}
}
```
## Any Questions? 
Kindly contact us at [limoxin@u.nus.edu](mailto:limoxin@u.nus.edu) for any issue. Thank you!





