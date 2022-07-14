Learning to Imagine: Integrating Counterfactual Thinking in Neural Discrete Reasoning
====================

This repositary contains the **TAT-HQA** dataset and the code for ACL 2022 paper, view [PDF](https://aclanthology.org/2022.acl-long.5.pdf).

You can download our [TAT-HQA dataset](https://github.com/NExTplusplus/TAT-HQA/tree/master/dataset_raw).

## Requirements
Our model is built upon `TagOp` ([Paper](https://aclanthology.org/2021.acl-long.254.pdf)) by further fine-tuning on TAT-HQA from the TagOp checkpoint. Please follow the process in [TAT-QA](https://github.com/NExTplusplus/TAT-QA) first and obtain a checkpoints for TagOp. 

## Training & Testing

### Preprocess Dataset

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_hqa/counter --output_dir tag_op/data_hqa/counter/roberta --encoder roberta --mode [train/dev]
```
The dataset_hqa folder contains heuristically-generated fields based on the raw data in dataset_raw, e.g. deriving operator, which is used in our implementation. 

### Training

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir tag_op/data_hqa/counter/roberta --save_dir tag_op/model_L2I --batch_size 32 --eval_batch_size 32 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-5  --weight_decay 5e-5 --seed 1 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-6 --bert_weight_decay 0.01 --log_per_updates 100 --eps 1e-6  --encoder roberta --test_data_dir tag_op/data_hqa/counter/roberta/ --roberta_model dataset_tagop/roberta.large --model_finetune_from path_to_tagop_checkpoint
```

Please fill in the --roberta_model with the path to the pre-trained `RoBERTa` model and --model_finetune_from with the TagOp checkpoint. 

Note that the current version only support training on TAT-HQA. If you would like to add TAT-QA data in training, you can do slight modification in tag_op/tagop/modeling_tagop_L2I.py .

### Testing
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/predictor.py --data_dir tag_op/data_hqa/counter/roberta --save_dir tag_op/model_counter_ft_layer_3_self_1_share_1_seed_123 --eval_batch_size 32 --model_path tag_op/model_L2I --encoder roberta --test_data_dir tag_op/data_hqa/counter/ --roberta_model path_to_roberta_model 
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





