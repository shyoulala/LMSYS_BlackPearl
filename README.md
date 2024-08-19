
## Hardware Specifications
- **CPU Cores**: 128
- **Memory**: 768 GB
- **GPU**: NVIDIA Tesla A100 80G
- **Number of GPUs**: 8
- - **OS/Platform**: Linux
## Third-Party Software
- **Python**: 3.10.14
- **PyTorch**: 2.3.1+cu121
- **CUDA**: 12.2
- **cuDNN**: 8.9.2.26
# Installation python packages
```bash
pip install -r requirements.txt
```

# Explanation of running the project
I have provided the complete training process (train model script), the distillation-only training process (fast train script), and the direct prediction process (Direct prediction). I have pre-placed the required datasets in the ./data directory. This dataset is consistent with the official one and comes from open-source datasets. For more details, please refer to the Model SUMMARY proposal I submitted.


# Explanation of directory tree
```
./model_path # pretrained model path
./src_fast  # fast train script (see fast train script section)
./src # The complete process of my solution
./data # train data and other data
./data/oof 
./data/processed_data
./data/processed_data/orgemma2fold4  # The training set with 70b probabilities can be directly distilled.
./data/processed_data/orgemma2fold2  # same 
./data/processed_data/orgemma2fold0  # same 
./data/processed_data/orgemma2fold1  # same 
./data/processed_data/orgemma2fold3  # same 
./data/lmsys-chatbot-arena
./sub # output dir
./model_save # save path for train model
./model_save_or
./model_save_or/v7_ut_gemma_v7_64r128_ddgemma2_16bit # post-pretrain ut model of gemma2-9b

```

# Model Download Preparation
Download these three models to the model_path folder:<br>
Llama3 70b (https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) (rename as llama3_70b)<br>
Qwen2 72b (https://huggingface.co/Qwen/Qwen2-72B-Instruct) (rename as qwen2_72b)<br>
Gemma2-9b (https://huggingface.co/google/gemma-2-9b-it) (rename as Gemma2_9b)


# train model script
（
The time estimates are based on a single machine with 8 NVIDIA A100 80GB GPUs）
1. **Preprocess Data** (Takes 2 hours)
    ```bash
    cd ./src
    sh run_data.sh
    ```
Convert the data into a format suitable for training.
2. **Post-Pretrain**
    ```bash
    sh run_post_pretrain_ut_three_model.sh
    ```
Use UT data to perform post-pretraining on three models, which takes 2.5 days.
3. **Fine-Tune Models for 5-Fold Results**
    ```bash
    sh run_pipeline.sh 0
    sh run_pipeline.sh 1
    sh run_pipeline.sh 2
    sh run_pipeline.sh 3
    sh run_pipeline.sh 4
    ```
Regarding distillation, the losses we use are as follows:
```python
loss_fun = nn.CrossEntropyLoss()
divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
cos_loss_fn = nn.CosineEmbeddingLoss()
outputs = model(batch['input_ids'], use_cache=False) # predict gemma2
logits = outputs.logits
grads = batch['grads']
grads1 = batch['grads'][:, :3] # qwen2 
grads2 = batch['grads'][:, 3:] # llama3
labels = batch['labels']
loss_ce = loss_fun(logits, labels)
loss_grad1 = divergence_loss_fn(
    F.log_softmax(logits / T, dim=1),
    F.softmax(grads1 / T, dim=1)
)
cos_loss1 = cos_loss_fn(F.softmax(grads1 / T, dim=1), F.softmax(logits / T, dim=1),
                        torch.ones(logits.size()[0]).to(logits.device))

loss_grad2 = divergence_loss_fn(
    F.log_softmax(logits / T, dim=1),
    F.softmax(grads2 / T, dim=1)
)
cos_loss2 = cos_loss_fn(F.softmax(grads2 / T, dim=1), F.softmax(logits / T, dim=1),
                        torch.ones(logits.size()[0]).to(logits.device))

loss = (loss_ce + loss_grad1 + cos_loss1 + loss_grad2 + cos_loss2) / 5.
```

Each fold includes training the Llama3 and Qwen2 models, predicting to obtain the probability distribution for the training set, and finally fine-tuning the Gemma-2b model.
4. **Merge LoRA and Quantize**
    ```bash
    sh run_final.sh
    ```
Here, the LoRA layers of the 5-fold Gemma2-9b models are merged and then quantized to 8-bit using GPTQ.
5. **Predict Test Set**
    ```python
    python predict_test.py
    ```
Once the previous steps are completed, you can directly run this script to make predictions.The final results will be saved in ./sub/submission.csv
If there is a new test set, you can directly replace ./data/lmsys-chatbot-arena/test.csv for prediction.<br>


# Direct prediction
If you want to directly predict using my best model, please first download the model (https://www.kaggle.com/datasets/sayoulala/v7-dpo-16bit-01234-8bit-all/) to the model_save folder and rename the folder to final_model_gptq. Then replace the file ./data/lmsys-chatbot-arena/test.csv and run the following script. This script is consistent with the Kaggle online inference script and will use GPU 0 and GPU 1.<br>
    ```
    python predict_test.py
    ```

# fast train script
Since training the 70b large model is extremely slow, I have provided the samples before distillation (with the probability distribution to be distilled already predicted). This way, you can directly distill and train the Gemma2-9b model.
```bash
cd ./src_fast
sh run_pipeline_fast.sh 0
sh run_pipeline_fast.sh 1
sh run_pipeline_fast.sh 2
sh run_pipeline_fast.sh 3
sh run_pipeline_fast.sh 4
sh run_final.sh
python predict_test.py
```
