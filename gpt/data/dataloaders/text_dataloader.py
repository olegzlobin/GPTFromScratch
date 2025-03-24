import os
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.file_path = file_path
        self.total_chars = os.path.getsize(file_path)

    def __len__(self):
        return self.total_chars // (self.block_size * 4)

    def __getitem__(self, idx):
        with open(self.file_path, "r", encoding="utf-8") as f:
            f.seek(idx * self.block_size * 4)
            text = f.read(self.block_size * 4)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 2:
            tokens = [self.tokenizer.eos_token_id] * 2

        start = random.randint(0, max(0, len(tokens) - self.block_size))
        input_ids = tokens[start:start + self.block_size]
        labels = tokens[start + 1:start + self.block_size + 1]

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels)
        }

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids = pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True,
            padding_value=pad_token_id
        )

        labels = pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True,
            padding_value=-100
        )

        return {
            'input_ids': input_ids,
            'labels': labels
        }

    def get_collate_fn(self):
        return lambda batch: self.collate_fn(
            batch=batch,
            pad_token_id=self.tokenizer.pad_token_id
        )