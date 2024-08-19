#!/usr/bin/env python


import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pickle
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import sys

model_path = sys.argv[1]
save_name = sys.argv[2]
print("model_path:", model_path)
print("save_name:", save_name)

MODEL_NAME = model_path
MAX_LENGTH = 1024
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id

# In[3]:


train = pd.read_parquet("../data/ultrafeedback_lmsysformat.parquet")

# In[4]:


train['prompt'] = train['prompt'].apply(lambda x: str([x]))

# In[5]:


train['prompt'] = train['prompt'].apply(lambda x: "\n".join(eval(x)))


# In[6]:


def do(row):
    # print(row)
    assert row['winner_tie'] + row['winner_model_a'] + row['winner_model_b'] == 1
    if row['winner_tie'] == 1:
        return "C"
    elif row['winner_model_a'] == 1:
        return "A"
    else:
        return "B"


train['label'] = train.apply(lambda row: do(row), axis=1)
train['label'].value_counts()


# In[7]:


def do(x):
    try:
        null = 'nan'
        return "\n".join(eval(x))
    except:
        x = x.replace("[", "").replace("]", "").strip()
        return x


train['response_a_str'] = train['response_a'].apply(lambda x: do(x))
train['response_b_str'] = train['response_b'].apply(lambda x: do(x))


# In[8]:


def create_rounds(query, answer_a, answer_b):
    prompt = f"""User question:
    \"""{query}\"""
    Answer A:
    \"""{answer_a}\"""
    Answer B:
    \"""{answer_b}\"""
    """
    return prompt


# In[9]:


texts = []
texts_token_len = []
for _, row in tqdm(train.iterrows()):
    query = ' '.join(row['prompt'].split(' ')[:256])
    answer_a = ' '.join(row['response_a_str'].split(' ')[:700])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:700])
    prompt_len = 256
    query_len = len(tokenizer.encode(query))
    answer_a_len = len(tokenizer.encode(answer_a))
    answer_b_len = len(tokenizer.encode(answer_b))
    if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
        query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
            tokenizer.encode(query)[:prompt_len])
        query_len = len(tokenizer.encode(query))
        if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
            remain_len = MAX_LENGTH - query_len
            token_answer_a = tokenizer.encode(answer_a)
            token_answer_b = tokenizer.encode(answer_b)
            if len(token_answer_a) > len(token_answer_b):
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
                        token_answer_b):
                    token_answer_a = token_answer_a[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            else:
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
                        token_answer_a):
                    token_answer_b = token_answer_b[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            answer_a = tokenizer.decode(token_answer_a)
            answer_b = tokenizer.decode(token_answer_b)
    prompt = create_rounds(query, answer_a, answer_b)
    texts.append(prompt)
    texts_token_len.append(len(tokenizer.encode(prompt)))
train['text'] = texts

# In[10]:


train2 = train.copy()


def do(row):
    # print(row)
    assert row['winner_tie'] + row['winner_model_a'] + row['winner_model_b'] == 1
    if row['winner_tie'] == 1:
        return "C"
    elif row['winner_model_a'] == 1:
        return "B"
    else:
        return "A"


train2['label'] = train2.apply(lambda row: do(row), axis=1)
print(train2['label'].value_counts())

texts = []
texts_token_len = []
for _, row in tqdm(train2.iterrows()):
    query = ' '.join(row['prompt'].split(' ')[:256])
    answer_a = ' '.join(row['response_a_str'].split(' ')[:700])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:700])
    prompt_len = 256
    query_len = len(tokenizer.encode(query))
    answer_a_len = len(tokenizer.encode(answer_a))
    answer_b_len = len(tokenizer.encode(answer_b))
    if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
        query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
            tokenizer.encode(query)[:prompt_len])
        query_len = len(tokenizer.encode(query))
        if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
            remain_len = MAX_LENGTH - query_len
            token_answer_a = tokenizer.encode(answer_a)
            token_answer_b = tokenizer.encode(answer_b)
            if len(token_answer_a) > len(token_answer_b):
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
                        token_answer_b):
                    token_answer_a = token_answer_a[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            else:
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
                        token_answer_a):
                    token_answer_b = token_answer_b[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            answer_a = tokenizer.decode(token_answer_a)
            answer_b = tokenizer.decode(token_answer_b)
    prompt = create_rounds(query, answer_b, answer_a)
    texts.append(prompt)
    texts_token_len.append(len(tokenizer.encode(prompt)))
train2['text'] = texts


# In[11]:


def do(x):
    if x == "C":
        return 2
    elif x == "B":
        return 1
    else:
        return 0

train['label'] = train['label'].apply(lambda x: do(x))
train2['label'] = train2['label'].apply(lambda x: do(x))



train = train.sample(frac=1., random_state=2023)
train2 = train2.sample(frac=1., random_state=2023)



train = train.drop_duplicates("id")
train2 = train2.drop_duplicates("id")



train_all = pd.concat([train, train2], axis=0)


train_all['text_len'] = train_all['text'].apply(lambda x: len(x.split(' ')))



train_all = train_all[train_all['text_len'] < 1200]

print(train_all['label'].value_counts())

with open(
        f"../data/processed_data/ut_{save_name}_train.parquet",
        'wb') as f:
    pickle.dump(train_all, f)

with open(
        f"../data/processed_data/ut_{save_name}_dev.parquet",
        'wb') as f:
    pickle.dump(train_all.sample(n=100), f)