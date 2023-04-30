Learning to Imagine: Integrating Counterfactual Thinking in Neural Discrete Reasoning
====================
**2022-12-13 Update:** This repo is currently under construction and will be ready soon. 

This repositary contains the **TAT-HQA** dataset and the code for ACL 2022 paper, view [PDF](https://aclanthology.org/2022.acl-long.5.pdf).

# Dataset 

The hypothetical questions in TAT-HQA are created from the factual questions of [TAT-QA](https://github.com/NExTplusplus/TAT-QA). 

[dataset_raw](https://github.com/NExTplusplus/TAT-HQA/tree/main/dataset_raw) contains a mixture of TAT-QA and TAT-HQA data, where TAT-HQA is annotated with `counterfactual` in the `answer_type` and the corresponding original TAT-QA question is recorded in `rel_question`. 
Similar to TAT-QA, the data files are organized by tables and a list of following passages, with a list of questions under each table. Refer to our [website](https://nextplusplus.github.io/TAT-HQA/) for detailed description of the data format. 

The questions contain the following keys, 
- `uid`: the unique question id.
- `order`: the order in the question list under the table and passages. 
- `question`: a question string.
- `answer`: a list of answer strings. The model is expected to predict all the answers. 
- `derivation`: for arithmetic questions, an equation of the answer calculation process. 
- `answer_type`: 5 types, span, multi-span, arithmetic, count or counterfactual. 
- `answer_from`: 3 types, table, text or table-text. 
- `rel_paragraph`: the order(s) of the relevant passage(s).
- `req_comparison`: True or False, whether the arithmetic question requires comparison. 
- `scale`: the model is also expected to predict a correct scale ('', thousand, million, billion, or percent) for each question. If the scale prediction is incorrect, the answer is evaluated as incorrect. 
- `rel_question (for hypothetical questions)`: the order of the corresponding factual question. Usually, it is the previous question. 

For our implementation of the paper method, we pre-process dataset_raw to generate some extra_fields. The `facts` and `mapping` are generated in the same way as TAT-QA (used for training the TagOP baseline). Apart from these, we extract the assumption substring from the hypothetical question (`question_if_part`), and we heuristically generate the `if_op` (SWAP, ADD, MINUS, DOUBLE, INCREASE_PERC, etc.) and `if_tagging` (the operands of if_op) for the Learning-to-Imagine module. The preocessed data is stored in [dataset_extra_field] and splitted by TAT-QA and TAT-HQA, saved in `orig` and `counter`. 

The test data of TAT-HQA is stored in `dataset_test_hqa/tathqa_dataset_dev.json`. 

## Requirements
The paper method is built upon `TagOp` ([Paper](https://aclanthology.org/2021.acl-long.254.pdf)) and further fine-tune on TAT-HQA from the TagOp checkpoint. Please follow the process in [TAT-QA](https://github.com/NExTplusplus/TAT-QA) to create the conda environment and download roberta.large. 
Our cuda version is 10.2, cuda driver version 440.33.01. We use one 32GB GPU. 

## Training & Testing

### Preprocess Dataset

Process the training/validation data by specifying 'counter' for TAT-HQA, 'orig' for TAT-QA, and 'both' for a mix of them.  

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_extra_field/[counter/orig/both] --output_dir tag_op/data/[counter/orig/both] --encoder roberta --mode [train/dev] --roberta_model roberta.large
```
Process the test data of TAT-HQA. 
```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_test_hqa --output_dir tag_op/data/test --encoder roberta --mode dev --data_format tathqa_dataset_{}.json --roberta_model roberta.large
```

### Training

First, we obtain a base model trained on a mix of TAT-QA&HQA data by running the following command. Set --roberta_model as the path to the roberta model. 

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir tag_op/data/both --save_dir tag_op/model_L2I --batch_size 32 --eval_batch_size 32 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4  --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 100 --eps 1e-6  --encoder roberta --test_data_dir tag_op/data/both/ --roberta_model roberta.large --cross_attn_layer 0 --do_finetune 0
```
This checkpoint of `model_L2I` be obtained from [this link](https://drive.google.com/file/d/1VzYy1a_PbOUnqZZLNW58jpWTFL8PRc3B/view?usp=sharing)

To evaluate the dev performance on TAT-QA or TAT-HQA, run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/[counter/orig] --test_data_dir tag_op/data/[counter/orig] --save_dir tag_op/model_L2I --eval_batch_size 32 --model_path tag_op/model_L2I --encoder roberta --roberta_model roberta.large --cross_attn_layer 0
```

The predicted answer file for the validation set is saved at `tag_op/model_L2I/answer_dev.json`. 

Fine-tune TAT-HQA on with the L2I module by setting --do_finetune, --model_finetune_from tag_op/model_L2I and --cross_attn_layer 3

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir tag_op/data/counter --save_dir tag_op/model_counter_ft_from_L2I --batch_size 32 --eval_batch_size 32 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-5  --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-6 --bert_weight_decay 0.01 --log_per_updates 100 --eps 1e-6  --encoder roberta --test_data_dir tag_op/data/counter/ --roberta_model roberta.large --cross_attn_layer 3 --do_finetune 1 --model_finetune_from tag_op/model_L2I
```
This checkpoint of `model_counter_ft_from_L2I` be obtained from [this link]()

### Testing

To test the performance on the dev set for TAT-HQA, run the following command and obtain the prediction file at `tag_op/model_counter_ft_from_L2I/answer_dev.json`. 

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/counter --test_data_dir tag_op/data/counter --save_dir tag_op/model_counter_ft_from_L2I --eval_batch_size 32 --model_path tag_op/model_counter_ft_from_L2I --encoder roberta --roberta_model roberta.large --cross_attn_layer 3
```

To obtain the test result on TAT-HQA, run the following command and obtain the prediction file at `tag_op/model_counter_ft_from_L2I/answer_dev.json`. 
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/test --test_data_dir tag_op/data/test --save_dir tag_op/model_counter_ft_from_L2I --eval_batch_size 32 --model_path tag_op/model_counter_ft_from_L2I --encoder roberta --roberta_model roberta.large --cross_attn_layer 3
```


### Evaluation
To use the evaluation script `evaluate.py` to evaluate the validation result of TAT-HQA, try running

```bash
python evaluate.py dataset_extra_field/counter/tatqa_and_hqa_field_dev.json tag_op/model_counter_ft_from_L2I/answer_dev.json 0
```
the 1st argument is the gold data path, and the 2nd argument is the prediction file path. 


## Checkpoint

The trained checkpoint will be released very soon! 

## Citation 
Please kindly add the following citation if you find our work helpful. Thanks!
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





