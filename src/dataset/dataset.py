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
        input_ids = self.tokenizer.encode(str(text), return_tensors='pt')

        return {"input_ids": input_ids.to(self.device),
                "summary": sample.summary}
