#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import sklearn
import numpy as np
import pandas as pd
import time
import pickle
from threading import Thread
from transformers import AutoTokenizer, LlamaModel
from transformers import LlamaForSequenceClassification, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from threading import Thread
from tqdm.auto import tqdm
import torch.nn.functional as F
import sys

MODEL_NAME = sys.argv[1]
WEIGHTS_PATH = sys.argv[2]
train_path = sys.argv[3]
save_path = sys.argv[4]
MAX_LENGTH = 1024
BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id

with open(train_path, 'rb') as f:
    test = pickle.load(f)

test = test.reset_index(drop=True)
print(test.shape)

test['text_len'] = test['text'].apply(lambda x: len(x.split(' ')))

test = test.sort_values("text_len", ascending=False)


print(test['label'].value_counts())


# In[7]:


class SFTDataset(Dataset):
    """
    Dataset for ChatGLMSFT model

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer, max_prompt_len) -> None:
        super().__init__()
        self.input_ids_list = []
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.indexs = []
        for _, data in tqdm(dataset.iterrows()):
            text = data['text'].replace(tokenizer.eos_token, "<end>")
            input_ids = tokenizer(text)['input_ids']
            input_ids.append(tokenizer.eos_token_id)
            self.input_ids_list.append(input_ids)
            self.indexs.append(data['order_index'])

    def __len__(self):
        length = len(self.input_ids_list)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=torch.tensor(self.input_ids_list[idx],
                                           dtype=torch.long),
                    order_indexs=torch.tensor(self.indexs[idx],
                                              dtype=torch.long), )

    def collate_fn(self, instances):
        input_ids, order_indexs = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "order_indexs",))

        def padding(inputs, val=0):
            batch_size = len(inputs)
            max_len = max(map(len, inputs))
            # max_len=MAX_LENGTH
            res = torch.ones([batch_size, max_len], dtype=inputs[0].dtype) * val
            for i, res_data in enumerate(res):
                seq_len = len(inputs[i])
                res_data[:seq_len] = inputs[i]
            return res

        input_ids = padding(input_ids, self.tokenizer.eos_token_id)
        return dict(
            input_ids=input_ids.long(),
            order_indexs=torch.stack(order_indexs),
        )

def load_model(device):
    # bnb_config =  BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=torch.float16,
    #     bnb_8bit_use_double_quant=False)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model_0 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map=device)
    base_model_0.config.pad_token_id = tokenizer.pad_token_id

    # LoRa configuration
    peft_config = LoraConfig(
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
    # Load weights
    model_0.load_state_dict(torch.load(WEIGHTS_PATH), strict=False)
    model_0 = model_0.eval()
    model_0.print_trainable_parameters()
    return model_0


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def inference(df, model, device, batch_size=BATCH_SIZE):
    # batch_size = 1
    df = df.reset_index(drop=True)
    model.config.pad_token_id = tokenizer.eos_token_id
    dataset = SFTDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn,
                        drop_last=False)

    predicts = []
    for batch in tqdm(loader):
        batch = to_device(batch, device)
        with torch.no_grad():
                outputs = model(batch['input_ids'], use_cache=False)
                logits = outputs.logits
                # logits = F.softmax(logits, dim=-1)
                logits = logits.float()
        predicts.extend([a.reshape(1, -1) for a in logits.detach().cpu().numpy()])
    predicts = np.concatenate(predicts)
    df['winner_model_a'] = predicts[:, 0]
    df['winner_model_b'] = predicts[:, 1]
    df['winner_tie'] = predicts[:, 2]
    return df


from threading import Thread

use_device = [f'cuda:{i}' for i in [0, 1, 2, 3, 4, 5, 6, 7]]
models = []
for device in use_device:
    print(device)
    models.append(load_model(device))


st = time.time()

test = test.reset_index(drop=True)
N_SAMPLES = len(test)

# Split the data into two subsets
# test = test.sample(n=100,random_state=2023)
test['fold'] = list(range(len(test)))
test['fold'] = test['fold'] % len(use_device)

results = []


# Function to run inference in a thread
def run_inference(df, model, device):
    results.append(inference(df, model, device))


ts = []
for index, device in enumerate(use_device):
    t0 = Thread(target=run_inference, args=(test[test['fold'] == index], models[index], device))
    ts.append(t0)
for i in range(len(ts)):
    ts[i].start()
for i in range(len(ts)):
    ts[i].join()

data = pd.concat(results, axis=0)
print(f"Processing complete. Total time: {time.time() - st}")


with open(save_path,'wb') as f:
    pickle.dump(data, f)
