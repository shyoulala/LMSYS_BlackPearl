
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pickle
import torch
import sklearn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


# ## Configurations

# In[5]:


@dataclass
class Config:
    gemma_dir = '../model_save/final_model_gptq'
    max_length = 2000
    batch_size = 8
    device = torch.device("cuda")
    tta = False  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input


cfg = Config()
MAX_LENGTH = cfg.max_length

# In[6]:


tokenizer = GemmaTokenizerFast.from_pretrained(Config.gemma_dir)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
tokenizer.pad_token_id, tokenizer.eos_token_id


test = pd.read_csv('../data/lmsys-chatbot-arena/test.csv')
sample_sub = pd.read_csv('../data/lmsys-chatbot-arena/sample_submission.csv')


def do(x):
    try:
        null = 'nan'
        return "\n".join(eval(x))
    except:
        x = x.replace("[", "").replace("]", "")
        return x


test['prompt'] = test['prompt'].apply(lambda x: do(x))
test['response_a_str'] = test['response_a'].apply(lambda x: do(x))
test['response_b_str'] = test['response_b'].apply(lambda x: do(x))


def create_rounds(query, answer_a, answer_b):
    prompt = f"""User question:
    \"""{query}\"""
    Answer A:
    \"""{answer_a}\"""
    Answer B:
    \"""{answer_b}\"""
    """
    return prompt


texts = []
texts_token_len = []
for _, row in tqdm(test.iterrows()):
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
test['text'] = texts
test['texts_token_len'] = texts_token_len
test['reverse'] = False

test2 = test.copy()
texts = []
texts_token_len = []
for _, row in tqdm(test2.iterrows()):
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
test2['text'] = texts
test2['texts_token_len'] = texts_token_len
test2['reverse'] = True

test = pd.concat([test, test2], axis=0)

test['text_len'] = test['text'].apply(lambda x: len(x.split(' ')))
test = test.sort_values("text_len", ascending=False)
# display(test.head(5))


# # Tokenize

# In[10]:


def tokenize(
        tokenizer, texts, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):
    #     prompt = ["<prompt>: " + p for p in prompt]
    #     response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    #     response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    #     if spread_max_length:
    #         prompt = tokenizer(prompt, max_length=max_length//3, truncation=True, padding=False).input_ids
    #         response_a = tokenizer(response_a, max_length=max_length//3, truncation=True, padding=False).input_ids
    #         response_b = tokenizer(response_b, max_length=max_length//3, truncation=True, padding=False).input_ids
    #         input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
    #         attention_mask = [[1]* len(i) for i in input_ids]
    #     else:
    #         text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
    #         tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)
    #         input_ids = tokenized.input_ids
    #         attention_mask = tokenized.attention_mask
    res = []
    for text in texts:
        input_ids = tokenizer(text)['input_ids']
        input_ids.append(tokenizer.eos_token_id)
        res.append(input_ids)
    return res


# In[11]:

data = pd.DataFrame()
data["id"] = test["id"]
data['reverse'] = test['reverse']
data["input_ids"] = tokenize(tokenizer, test["text"])
data["length"] = data["input_ids"].apply(len)
data = data.reset_index(drop=True)
data['length'].describe([0.9])


data.shape

# In[13]:


print(data["input_ids"][0])
print(tokenizer.decode(data["input_ids"][0]))


import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import LlamaPreTrainedModel, LlamaModel, Gemma2PreTrainedModel, Gemma2Model, Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, List, Union, Tuple


class Gemma2ForSequenceClassificationV1(Gemma2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        print("v1:::")
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        #         logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1
        hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), sequence_lengths]  # eos
        pooled_logits = self.score(hidden_states)

        return pooled_logits


# In[15]:


print('4bit')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model on GPU 0
device_0 = torch.device('cuda:0')
model_0 = Gemma2ForSequenceClassificationV1.from_pretrained(
    cfg.gemma_dir,
    num_labels=3,
    device_map=device_0,
    #     torch_dtype=torch.float16,
    #     quantization_config=bnb_config,
    use_cache=False,
)
model_0.config.pad_token_id = tokenizer.pad_token_id

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
model_1 = Gemma2ForSequenceClassificationV1.from_pretrained(
    cfg.gemma_dir,
    num_labels=3,
    device_map=device_1,
    #     torch_dtype=torch.float16,
    #     quantization_config=bnb_config,
    use_cache=False,
)
model_1.config.pad_token_id = tokenizer.pad_token_id

# #### Load LoRA adapter

# In[16]:


model_1.config.pad_token_id, tokenizer.pad_token_id

# In[17]:


# peft_config = LoraConfig(
#         r=64,
#         lora_alpha=128,
#         target_modules=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#             "gate_proj",
#             "up_proj",
#             "down_proj",
#             "lm_head",
#             # "score",
#         ],
#         bias="none",
#         lora_dropout=0.05,  # Conventional
#         task_type="CAUSAL_LM",
#     )
# # Get peft
# model_0 = get_peft_model(model_0, peft_config).to(device_0)
# # Load weights
# model_0.load_state_dict(torch.load(Config.lora_dir1), strict=False)
model_0 = model_0.eval()

# model_1 = get_peft_model(model_1, peft_config).to(device_1)
# model_1.load_state_dict(torch.load(Config.lora_dir2), strict=False)
model_1 = model_1.eval()


# # Inference
#

# In[18]:


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win, tie = [], [], []

    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        #         attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.softmax(-1).cpu()

        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())

    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie

    return df


# In[19]:


st = time.time()

# sort by input length to fully leverage dynaminc padding
data = data.sort_values("length", ascending=False)
data = data.reset_index(drop=True)
# the total #tokens in sub_1 and sub_2 should be more or less the same
sub_1 = data[data['reverse'] == False]
sub_2 = data[data['reverse'] == True]
# sub_1 = data.iloc[0::2].copy()
# sub_2 = data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

result_df = pd.concat(list(results), axis=0)
# proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

print(f"elapsed time: {time.time() - st}")

# In[20]:


winner_model_a = result_df['winner_model_a'].values.copy()
winner_model_b = result_df['winner_model_b'].values.copy()
result_df.loc[result_df['reverse'].values, 'winner_model_a'] = winner_model_b[result_df['reverse'].values]
result_df.loc[result_df['reverse'].values, 'winner_model_b'] = winner_model_a[result_df['reverse'].values]

# In[21]:


temp = result_df.groupby('id')['winner_model_a'].mean().to_dict()
temp2 = result_df.groupby('id')['winner_model_b'].mean().to_dict()
temp3 = result_df.groupby('id')['winner_tie'].mean().to_dict()

# In[22]:


sample_sub['winner_model_a'] = sample_sub['id'].map(temp)
sample_sub['winner_model_b'] = sample_sub['id'].map(temp2)
sample_sub['winner_tie'] = sample_sub['id'].map(temp3)
sample_sub[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']].to_csv('../sub/submission.csv', index=False)
