import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pandas as pd


# text CNN net
class MultiCNNText(nn.Module):
    def __init__(self, opt, weight):
        super(MultiCNNText, self).__init__()
        self.model_name = 'MultiCNNTextBNDeep'
        self.embed=nn.Embedding.from_pretrained(weight)
        self.embed.weight.requires_grad = False
        kernel_sizes = [1, 2, 3]
        content_convs = [nn.Sequential(
            nn.Conv1d(in_channels=opt["embedding_dim"],
                      out_channels=opt["content_dim"],
                      kernel_size=kernel_size),
            nn.BatchNorm1d(opt["content_dim"]),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=opt["content_dim"],
                      out_channels=opt["content_dim"] * 2,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(opt["content_dim"] * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(opt["content_seq_len"] - kernel_size * 2 + 2))
        )
            for kernel_size in kernel_sizes]

        self.content_convs = nn.ModuleList(content_convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (opt["content_dim"]) * 2, opt["linear_hidden_size"]),
            # nn.Dropout(0.4, inplace=True),
            nn.BatchNorm1d(opt["linear_hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Linear(opt["linear_hidden_size"], 2)
        )
        self.dropout=nn.Dropout(opt['dropout'])

    def forward(self, content):
        content=self.embed(content)
        content_out = [content_conv(content.permute(0, 2, 1)) for content_conv in self.content_convs]
        conv_out = torch.cat((content_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        reshaped=self.dropout(reshaped)
        logits = self.fc((reshaped))
        return logits
