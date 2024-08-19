import time
from dataclasses import dataclass
import pickle
import torch
import sklearn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

lora_dir = '../model_save/gemma2fold0_16bit_load_fintune/best_val_loss_model/adapter.bin'
d1 = torch.load(lora_dir)
lora_dir = '../model_save/gemma2fold1_16bit_load_fintune/best_val_loss_model/adapter.bin'
d2 = torch.load(lora_dir)
lora_dir = '../model_save/gemma2fold2_16bit_load_fintune/best_val_loss_model/adapter.bin'
d3 = torch.load(lora_dir)
lora_dir = '../model_save/gemma2fold3_16bit_load_fintune/best_val_loss_model/adapter.bin'
d4 = torch.load(lora_dir)
lora_dir = '../model_save/gemma2fold4_16bit_load_fintune/best_val_loss_model/adapter.bin'
d5 = torch.load(lora_dir)

d = {}
for k, v in d1.items():
    v = d1[k] + d2[k] + d3[k] + d4[k] + d5[k]
    v = v / 5.
    d[k] = v
torch.save(d, "../model_save/final_adapter.bin")
