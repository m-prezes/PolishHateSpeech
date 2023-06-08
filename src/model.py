from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("sdadas/polish-distilroberta")
        self.herbert = AutoModel.from_pretrained("sdadas/polish-distilroberta")

        # for param in self.herbert.parameters():
        #     param.requires_grad = False
        
        self.linear = nn.Linear(768, 1)
    
    def forward(self, embeddings):
        output = self.herbert(**embeddings)
        output = output['pooler_output']
        output = self.linear(output)
        return nn.Sigmoid()(output)
    