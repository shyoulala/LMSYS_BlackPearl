#!/bin/bash
set -e

# gemma_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/gemma-2-9b-it
# gemma_path_ut=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96/workplace/LLM_notes_generate/tt/LMSYS_BlackPearl/model_save_or/v7_ut_gemma_v7_64r128_ddgemma2_16bit/epoch_0_model/adapter.bin


gemma_path=../model_path/Gemma2_9b
gemma_path_ut=../model_save_or/v7_ut_gemma_v7_64r128_ddgemma2_16bit/epoch_0_model/adapter.bin


fold=$1
echo run:${fold}
# 微调gemma-9b
sh run_fintune_16bit.sh orgemma2fold${fold} ${gemma_path} ${gemma_path_ut}

