#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import sys
from azureml.core import Run
import argparse
import mlflow
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, T5Tokenizer,
                          TextDataset, Trainer, TrainerCallback,
                          TrainingArguments, default_data_collator)

# 日本語対応
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 引数
parser = argparse.ArgumentParser()

parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--model_name_or_path', default='rinna/japanese-gpt2-medium')

args = parser.parse_args()

# Azure ML 事前準備
run = Run.get_context()
ws = run.experiment.workspace

# mlflow trackinr uri の設定
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# tokenizer, model オブジェクトのロード
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium", do_lower_case=True)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

# データセット
train_path = 'train.txt'
test_path = 'test.txt'

train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=512)
eval_dataset = TextDataset(tokenizer=tokenizer, file_path=test_path, block_size=512)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# mlflow でログを取るための callback クラス
class MyCallback(TrainerCallback):
    def __init__(self, azureml_run=None):
        self.mlflow = mlflow

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, v, step=state.global_step)

# Trainer 引数
training_args = TrainingArguments(
    output_dir="./outputs", 
    overwrite_output_dir=True, 
    max_steps=args.max_steps,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=50,
    fp16=True,
    report_to=["none"],
    ort=True,
    )


# モデル学習の設定
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback]
)

# モデル学習開始
trainer.train()

# モデルの保存
trainer.save_model()

# モデルの検証
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium", do_lower_case=True)
model = AutoModelForCausalLM.from_pretrained("./outputs")

input = tokenizer.encode("仕事", return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=100)
print(tokenizer.batch_decode(output))

input = tokenizer.encode("料理", return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=100)
print(tokenizer.batch_decode(output))


input = tokenizer.encode("握手をしたら、", return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=100)
print(tokenizer.batch_decode(output))
