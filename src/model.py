from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
        self.herbert = AutoModel.from_pretrained("allegro/herbert-base-cased")

        for param in self.herbert.parameters():
            param.requires_grad = False
        
        self.linear = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 1)
    
    def forward(self, embeddings):
        output = self.herbert(**embeddings)
        output = output['pooler_output']
        output = self.linear(output)
        output = nn.ReLU()(output)
        output = self.linear2(output)
        return nn.Sigmoid()(output)
    