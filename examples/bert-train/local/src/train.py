import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from . import datasets, models

# seed
pl.seed_everything(1234)

# device (cpu or gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 学習データのロード
df = pd.read_csv("./livedoor.tsv", delimiter='\t')
df = df.dropna()
df.head()

# 学習データとテストデータに分割
X_train, X_test = train_test_split(df, test_size=0.2, stratify=df['label_index'])
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

X_train.to_csv("./livedoor-train.tsv", sep='\t', index=False)
X_test.to_csv("./livedoor-test.tsv", sep='\t', index=False)


# Data Loader
train_dataset = datasets.LivedoorDataset(X_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = datasets.LivedoorDataset(X_train)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)


# 学習設定
bert_model = models.LitBert()

# 学習対象パラメータ
for param in bert_model.bert.bert.parameters():
    param.requires_grad = False

for param in bert_model.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

# モデル学習
trainer = pl.Trainer(gpus=1, default_root_dir='pl-model', max_epochs=20)
trainer.fit(bert_model, train_dataloader=train_loader, val_dataloaders=test_loader)

# 検証
result = trainer.test(bert_model, test_dataloaders=test_loader)
print(result)

# モデル保存
trainer.save_checkpoint("./outputs/models/bert-livedoor-epoch30.ckpt")



