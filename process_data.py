import os, enum, torch
from torch.utils.data import Dataset

DatasetSplit = enum.Enum('DatasetSplit', 'train valid test')

class SimpleTextDataset(Dataset):
    def __init__(self, paths, tokenizer, context, split: DatasetSplit,
                 splits=(0.9, 0.05, 0.05)):
        assert abs(sum(splits) - 1.0) < 1e-6
        self.context = context
        self.tokenizer = tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- 1) Tokenize all text into one long stream ---
        all_ids = []
        for p in (paths if isinstance(paths, (list, tuple)) else [paths]):
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
            if not txt.endswith(tokenizer.eos_token):
                txt += tokenizer.eos_token
            all_ids.extend(tokenizer(txt, add_special_tokens=False)["input_ids"])
        self.data = torch.tensor(all_ids, dtype=torch.long)

        # --- 2) Define split ranges ---
        N = len(self.data)
        n_train = int(splits[0] * N)
        n_valid = int(splits[1] * N)
        split_ranges = {
            DatasetSplit.train: (0, n_train),
            DatasetSplit.valid: (n_train, n_train + n_valid),
            DatasetSplit.test : (n_train + n_valid, N),
        }
        self.start, self.end = split_ranges[split]

        # Trim the split to a multiple of context (drop remainder)
        usable = (self.end - self.start) // context * context
        self.data = self.data[self.start : self.start + usable]

        # --- 3) Compute how many blocks we have ---
        self.n = len(self.data) // context

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.context
        end = start + self.context
        x = self.data[start:end]
        y = x  # GPT-2 internally shifts when computing the loss
        return x, y
