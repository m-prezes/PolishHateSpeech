import torch
from torch.utils.data import Dataset, DataLoader

class PolishHateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts.loc[index].values[0]
        label = self.labels.loc[index].values[0]

        embbedings = self.tokenizer(
            text,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        label = torch.tensor(label, dtype=torch.float)


        return embbedings, label