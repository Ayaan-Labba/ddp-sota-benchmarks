from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict

class GenerativeIEDataset(Dataset):
    """
    Dataset class for generative information extraction models.

    Args:
        data (List[Dict[str, any]]): List of data examples.
        tokenizer (AutoTokenizer): Tokenizer for text processing.
        prefix (str, optional): Prefix to add to each input text. Defaults to ''.
        max_source_length (int, optional): Maximum length for source sequences. Defaults to 512.

    Returns:
        Dataset: A PyTorch Dataset yielding tokenised input and labels (tokenised later by collate function).
    """
    def __init__(self, data: List[Dict[str, any]], tokenizer: AutoTokenizer, prefix: str = '', max_source_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_source_length = max_source_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        input_text = self.prefix + text
        tokenized = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized.input_ids.squeeze(0), # squeeze to remove batch dim added by tokenizer
            "attention_mask": tokenized.attention_mask.squeeze(0),
            "labels": text # placeholder, modify in collate function
        }