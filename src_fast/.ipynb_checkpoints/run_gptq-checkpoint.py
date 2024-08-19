import torch
import sklearn
import numpy as np
import pandas as pd
import time
import pickle
from threading import Thread
from transformers import AutoTokenizer, LlamaModel
from transformers import LlamaForSequenceClassification, BitsAndBytesConfig,AutoModelForSequenceClassification,Gemma2ForCausalLM
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader,Dataset
from threading import Thread
from tqdm.auto import tqdm
import torch.nn.functional as F


# MODEL_NAME = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/gemma-2-9b-it'
MODEL_NAME = "../model_path/Gemma2_9b"
WEIGHTS_PATH = '../model_save/final_adapter.bin'

def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model_0 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels =3,
        torch_dtype=torch.float16,
        device_map=device)
    peft_config= LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
                # "score",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
    model_0 = get_peft_model(base_model_0, peft_config).to(device)
    model_0.load_state_dict(torch.load(WEIGHTS_PATH,map_location=model_0.device), strict=False)
    model_0=model_0.eval()
    model_0.print_trainable_parameters()
    model_0 = model_0.merge_and_unload()
    return model_0,tokenizer

model,tokenizer = load_model('cuda:0')

model.save_pretrained("../model_save/final_model")
tokenizer.save_pretrained("../model_save/final_model")

print('save success!!!')
import torch
from transformers import GPTQConfig, AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM,Gemma2ForCausalLM
from peft import PeftModel
import logging
import pandas as pd
import pickle

import logging
logging.basicConfig(
   format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)
WEIGHTS_PATH = "../model_save/final_model"
QUANTIZED_MODEL_DIR = "../model_save/final_model_gptq"
tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_PATH)
def build_dataset():
    with open("../data/processed_data/orgemma2fold1/train.parquet",'rb') as f:
        train = pickle.load(f).sample(n=128, random_state=2023)
    return train

text_list = build_dataset()['text'].to_list()


gptq_config = GPTQConfig(bits=8, dataset = text_list,group_size=128,model_seqlen=2100,
                         # exllama_config = {"version":2},
                         # modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"],
                         #                               ['mlp.gate_proj','mlp.up_proj','mlp.down_proj']],
                         tokenizer=tokenizer)

merged_quantized_model = AutoModelForSequenceClassification.from_pretrained(
    WEIGHTS_PATH,
    torch_dtype=torch.float16,
    num_labels =3,
    device_map='cuda:0',
    quantization_config=gptq_config,
    )

merged_quantized_model.save_pretrained(QUANTIZED_MODEL_DIR)
tokenizer.save_pretrained(QUANTIZED_MODEL_DIR)