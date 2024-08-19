#!/bin/bash
set -e


qwen_path=../model_path/qwen2_72b
llama_path=../model_path/llama3_70b
gemma_path=../model_path/Gemma2_9b


# qwen_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96/zy_model_path/Qwen2-72B-Instruct
# llama_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/llama3-70B
# gemma_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/gemma-2-9b-it

sh run_post_pretrain.sh llama3 ${llama_path}
sh run_post_pretrain.sh qwen2 ${qwen_path}
sh run_post_pretrain.sh gemma2 ${gemma_path}






