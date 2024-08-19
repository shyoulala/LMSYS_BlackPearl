#!/bin/bash

#### input
name=$1
MODEL_PATH=$2
lora_path=$3
fold=$4
model_name="load_fintune"
MODEL_USE="4bit"
echo ${name}
echo ${MODEL_PATH}
echo ${lora_path}

DATA_DIR=../data/processed_data/
ZERO_STAGE=2
OUTPUT=../model_save/${name}_${MODEL_USE}_${model_name}
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}
MASTER_PORT=20345
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1,2,3,4,5,6,7 train_ce_model_nice_memory.py \
       --project_name ${name}_${MODEL_USE} \
       --lora_path ${lora_path} \
       --model_name ${model_name} \
       --train_dataset_path ${DATA_DIR}${name}fold${fold}/train.parquet \
       --dev_dataset_path  ${DATA_DIR}${name}fold${fold}/dev.parquet \
       --model_name_or_path ${MODEL_PATH} \
       --per_device_train_batch_size 8 \
       --per_device_eval_batch_size 8 \
       --gradient_accumulation_steps 1 \
       --max_prompt_len 1024 \
       --max_completion_len 256 \
       --earystop 0 \
       --save_batch_steps 200 \
       --eary_stop_epoch 5 \
       --save_per_epoch 1 \
       --num_train_epochs 2  \
       --debug_code 0 \
       --learning_rate 1e-5 \
       --num_warmup_steps 100 \
       --weight_decay 0. \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing