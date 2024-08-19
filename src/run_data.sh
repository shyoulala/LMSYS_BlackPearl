#!/bin/bash


# qwen_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96/zy_model_path/Qwen2-72B-Instruct
# llama_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/llama3-70B
# gemma_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/gemma-2-9b-it


qwen_path=../model_path/qwen2_72b
llama_path=../model_path/llama3_70b
gemma_path=../model_path/Gemma2_9b


python prepare_data.py ${qwen_path} qwen2
python prepare_data.py ${llama_path} llama3
python prepare_data.py ${gemma_path} gemma2




python prepare_data_ut.py ${qwen_path} qwen2
python prepare_data_ut.py ${llama_path} llama3
python prepare_data_ut.py ${gemma_path} gemma2

