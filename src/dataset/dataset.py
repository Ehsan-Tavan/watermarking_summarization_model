# ============================ Third Party libs ============================
from typing import List, Optional
import torch

# ============================ My packages ============================
from src.utils import InputExample


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: List[InputExample],
                 tokenizer,
                 max_len: int = 50,
                 device: str = "cpu"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item_index):
        sample = self.data[item_index]
        text = sample.text
        tokenized_sample = self.tokenizer.encode_plus(text=str(text),
                                                      max_length=self.max_len,
                                                      return_tensors="pt",
                                                      padding=True,
                                                      truncation=True)

        # input_ids = tokenized_sample["input_ids"].flatten()
        # attention_mask = tokenized_sample["attention_mask"].flatten()

        return {"input_ids": tokenized_sample["input_ids"].to(self.device),
                "summary": sample.summary}
