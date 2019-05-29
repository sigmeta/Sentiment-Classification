import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pandas as pd


# text LSTM net
class LSTMText(nn.Module):
    def __init__(self, opt, weight):
        super(LSTMText, self).__init__()
        self.model_name = 'LSTMTextBNDeep'
        self.embed=nn.Embedding.from_pretrained(weight)
        self.embed.weight.requires_grad = False
        self.lstm=nn.LSTM(input_size=opt["embedding_dim"],hidden_size=opt["content_dim"]//2,
                          num_layers=1,batch_first=True,dropout=opt['dropout'],
                          bidirectional=True)
        self.dense=nn.Linear(opt['content_dim'],64)
        self.out=nn.Linear(64,2)
        self.dropout=nn.Dropout(opt['dropout'])


    def forward(self, content):
        content=self.embed(content)
        content_out,(hidden,cell) = self.lstm(content)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden=hidden.view(hidden.size(0),-1)
        #print(hidden.size())
        hidden=self.dropout(hidden)
        out = self.dense(hidden)
        out=self.dropout(out)
        logits=self.out(out)
        return logits
