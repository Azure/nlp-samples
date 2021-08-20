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

# ライブラリのインポート
import io
import sys
from azureml.core import Run, Dataset
import argparse
import mlflow
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          T5Tokenizer,
                          Trainer, TrainerCallback,
                          TrainingArguments)

# 日本語対応
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 引数
parser = argparse.ArgumentParser()
parser.add_argument('--num_train_epochs', type=int, default=1)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--model_name_or_path', default='rinna/japanese-gpt2-medium')
args = parser.parse_args()


# パラメータ
block_size = 128

# mlflow でログを取るための callback クラス
class MyCallback(TrainerCallback):
    def __init__(self, azureml_run=None):
        self.mlflow = mlflow

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, v, step=state.global_step)

# tokenize のメソッド
def tokenize_function(examples):
    return tokenizer(examples["text"])

# データを連結して blocksize ごとに保持
def group_texts(examples):
    # Concatenate all texts.
    # 補足 : keys は train, test, validation の 3 個の key が入っている
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()} #[[]] -> []
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Job start ...

# Azure ML の Run と Workspace オブジェクトの取得
run = Run.get_context()
ws = run.experiment.workspace

# mlflow trackinr uri の設定
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


# データセットの準備
dataset = Dataset.get_by_name(ws, name='train')
dataset.download(target_path='.', overwrite=False)

dataset = Dataset.get_by_name(ws, name='test')
dataset.download(target_path='.', overwrite=False)

train_path = 'train.txt'
test_path = 'test.txt'

datasets = load_dataset("text", data_files={"train": 'train.txt', "validation": 'test.txt'})


# データ前処理
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=True, do_lower_case=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
print(tokenizer.decode(tokenized_datasets["train"][1]['input_ids']))


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

assert len(lm_datasets["train"][0]["attention_mask"]) == block_size

print(tokenizer.decode(lm_datasets["train"][0]["input_ids"]))
print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))


# 学習済みモデル試行
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
input = tokenizer.encode("こんにちは、", return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=64, num_return_sequences=100)
print(tokenizer.batch_decode(output))


# Trainer の設定
model_name = args.model_name_or_path.split("/")[-1]

training_args = TrainingArguments(
    output_dir="outputs", 
    overwrite_output_dir=True, 
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=100,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=args.num_train_epochs,
    report_to=["none"],
    ort=True,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    callbacks=[MyCallback]
)

# モデル学習開始
with mlflow.start_run() as run:  
    trainer.train()
    
    # モデルの保存
    trainer.save_model("outputs/models")

    # mlflow api でのモデル登録
    model_uri = run.info.artifact_uri + '/outputs/models'
    mlflow.log_artifacts("outputs/models", artifact_path="outputs/models") 
    mlflow.register_model(model_uri, "test-model")


# モデルのテスト
model = AutoModelForCausalLM.from_pretrained("outputs/models")
input = tokenizer.encode("こんにちは、", return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=64, num_return_sequences=100)
print(tokenizer.batch_decode(output))
