from transformers import BertJapaneseTokenizer
import torch
from torch.utils.data import Dataset


class LivedoorDataset(Dataset):
    def __init__(self, data):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.text = list(data['text'])
        self.label = list(data['label_index'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]
        inputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', max_length=512, truncation=True)        

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.long)
        }

