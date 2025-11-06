import os, enum, torch, random
from torch.utils.data import Dataset, DataLoader, Subset

DatasetSplit = enum.Enum('DatasetSplit', 'train valid test')

class SimpleTextDataset(Dataset):
    def __init__(self, paths, tokenizer, context, splits=(0.9, 0.05, 0.05), equalize=True):
        assert abs(sum(splits) - 1.0) < 1e-6
        self.context = context
        self.tokenizer = tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.eos_token is not None, "Tokenizer must have an eos_token."

        # --- Tokenize and combine ---
        pieces = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                txt = f.read()
            if not txt.endswith(tokenizer.eos_token):
                txt += tokenizer.eos_token
            ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
            pieces.append(ids)

        # --- Equalize across files ---
        if equalize:
            target = min(len(p) for p in pieces)
            target = (target // context) * context
            if target == 0:
                raise ValueError("equalize=True but one file is smaller than context.")
            pieces = [p[:target] for p in pieces]

        flat = [tok for p in pieces for tok in p]
        usable = len(flat) // context * context
        self.data = torch.tensor(flat[:usable], dtype=torch.long)
        self.n = len(self.data) // context

        # --- Compute split indices ---
        N = self.n
        n_train = int(splits[0] * N)
        n_valid = int(splits[1] * N)
        idxs = {
            DatasetSplit.train: range(0, n_train),
            DatasetSplit.valid: range(n_train, n_train + n_valid),
            DatasetSplit.test:  range(n_train + n_valid, N),
        }
        self.splits = idxs  # store for Subset creation

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.context
        e = s + self.context
        x = self.data[s:e]
        y = x
        return x, y

    def get_split(self, split: DatasetSplit):
        """Return a Subset corresponding to the chosen split."""
        return Subset(self, list(self.splits[split]))