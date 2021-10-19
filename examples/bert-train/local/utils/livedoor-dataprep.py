# livedoor ニュースのデータ前処理
# 参考サイト: https://qiita.com/sugulu_Ogawa_ISID/items/697bd03499c1de9cf082

import tarfile
import os
import numpy as np
import urllib.request

DOWNLOAD = True
EXTRACT = True
DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FILENAME = "ldcc-20140209.tar.gz"

os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

if DOWNLOAD is True:
    # ダウンロード
    text_url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    urllib.request.urlretrieve(text_url, os.path.join(RAW_DIR, FILENAME))

if EXTRACT is True:
    # gz ファイルを解凍します。
    os.makedirs(os.path.join(INTERIM_DIR, 'livedoor'),exist_ok=True)
    with tarfile.open(os.path.join(RAW_DIR, FILENAME), 'r:gz') as tar:
        tar.extractall(path=os.path.join(INTERIM_DIR, 'livedoor'))
        tar.close()

files_folders = [name for name in os.listdir(os.path.join(INTERIM_DIR, 'livedoor'))]
print(files_folders)


categories = ["dokujo-tsushin", "it-life-hack", "kaden-channel", "livedoor-homme",
        "movie-enter", "peachy", "smax", "sports-watch", "topic-news"]


def extract_main_txt(file_name):
    with open(file_name) as text_file:
        text = text_file.readlines()[3:]
        text = [sentence.strip() for sentence in text] # 空白文字の削除
        text = list(filter(lambda line: line != '', text))
        text = ''.join(text)
        text = text.translate(str.maketrans(
            {'\n': '', '\t': '', '\r': '', '\u3000': ''}))  # 改行やタブ、全角スペースを消す
        return text


import glob

list_text = []
list_label = []

os.makedirs(os.path.join(INTERIM_DIR, 'livedoor/text'),exist_ok=True)
for cat in categories:
    text_files = glob.glob(os.path.join(os.path.join(INTERIM_DIR, 'livedoor/text'), cat, "*.txt"))
    body = [extract_main_txt(text_file) for text_file in text_files]
    label = [cat] * len(body)
    list_text.extend(body)
    list_label.extend(label)

print(list_text[0])
print(list_label[0])

import pandas as pd

df = pd.DataFrame({'text': list_text, 'label': list_label})
df = df.dropna()
print(df.shape)
df.head()


# カテゴリーの辞書を作成

id2cat = dict((id,cat) for id, cat in enumerate(categories))
cat2id = dict((id2cat[key], key) for key in id2cat)
print(id2cat, cat2id)

np.save(os.path.join(PROCESSED_DIR, "id2cat.npy"), id2cat)
np.save(os.path.join(PROCESSED_DIR, "cat2id.npy"), cat2id)

df["label_index"] = df["label"].map(cat2id)
df.head()

df = df.loc[:, ["text", "label_index", "label"]]
df.head()

# 順番をシャッフルする
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()

# シャッフル済みの全件データを保存
df.to_csv(os.path.join(PROCESSED_DIR, 'livedoor.tsv'), sep='\t', index=False)
