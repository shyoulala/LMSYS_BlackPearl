#!/bin/bash
set -e

# qwen_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96/zy_model_path/Qwen2-72B-Instruct
# llama_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/llama3-70B
# gemma_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/gemma-2-9b-it



qwen_path=../model_path/qwen2_72b
llama_path=../model_path/llama3_70b
gemma_path=../model_path/Gemma2_9b

qwen_path_ut=../model_save/qwen2_4bit_pretrain/epoch_0_model/adapter.bin
llama_path_ut=../model_save/llama3_4bit_pretrain/epoch_0_model/adapter.bin
gemma_path_ut=../model_save/gemma2_4bit_pretrain/epoch_0_model/adapter.bin


fold=$1
echo run:${fold}
# train llama3 70b
sh run_fintune.sh llama3 ${llama_path}  ${llama_path_ut} ${fold}
# predict train logits
python predict_train.py ${llama_path} ../model_save/llama3_4bit_load_fintune/epoch_0_model/adapter.bin ../data/processed_data/llama3fold${fold}/train.parquet ../data/oof/llama3fold${fold}_train.parquet

# train qwen2 70b
sh run_fintune.sh qwen2 ${qwen_path}  ${qwen_path_ut} ${fold}
# predict train logits
python predict_train.py ${qwen_path} ../model_save/qwen2_4bit_load_fintune/epoch_0_model/adapter.bin ../data/processed_data/qwen2fold${fold}/train.parquet ../data/oof/qwen2fold${fold}_train.parquet

# merge  logits 
python merge_logits.py ../data/processed_data/gemma2fold${fold}/train.parquet ../data/oof/qwen2fold${fold}_train.parquet ../data/oof/llama3fold${fold}_train.parquet ../data/processed_data/gemma2fold${fold}/train_logits.parquet

# distill fintune gemma2-9b
sh run_fintune_16bit_distill.sh gemma2 ${gemma_path} ${gemma_path_ut} ${fold}
