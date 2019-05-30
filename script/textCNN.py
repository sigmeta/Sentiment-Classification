import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch.utils.data
from torch.autograd import Variable
from gensim.models import Word2Vec
from model.textCNN import MultiCNNText
from model.LSTM import LSTMText
import json
from tqdm import tqdm,trange


class DataProcessor():
    def __init__(self):
        self.train_path = '../data/train.json'
        self.test_path = '../data/test.json'
        self.max_seq_len = 512
        self.batch_size = 128

    def get_embedding(self, wv_path="../data/word2vec.model"):
        extra_tokens=['<PAD>','<UNK>']
        wv=Word2Vec.load(wv_path)
        self.word2id={w:i+len(extra_tokens) for i,w in enumerate(wv.wv.vocab)}
        self.id2word={i+len(extra_tokens):w for i,w in enumerate(wv.wv.vocab)}
        for i,w in enumerate(extra_tokens):
            self.word2id[w]=i
            self.id2word[i]=w

        weight=np.zeros((len(self.word2id)+len(extra_tokens),wv.wv.vector_size))
        for i,w in enumerate(wv.wv.vocab):
            weight[i+len(extra_tokens),:]=wv.wv[w]
        return torch.FloatTensor(weight)

    def tokenizer(self, sentence):
        pass

    def get_train_data(self):
        with open(self.train_path, encoding='utf8') as f:
            tr = json.loads(f.read())
        train = torch.zeros(len(tr), self.max_seq_len, dtype=torch.long)
        target = torch.zeros(len(tr), dtype=torch.long)
        for i, d in enumerate(tr):
            target[i] = d['label']
            sentence = d['text'][:self.max_seq_len]
            for j, word in enumerate(sentence):
                if word not in self.word2id:
                    word='<UNK>'
                train[i,j]=self.word2id[word]
        train_dataset=TensorDataset(train,target)
        train_sampler=RandomSampler(train_dataset)
        train_dataloader=DataLoader(train_dataset,sampler=train_sampler,batch_size=self.batch_size)
        return train_dataloader

    def get_test_data(self):
        with open(self.test_path, encoding='utf8') as f:
            te = json.loads(f.read())
        test = torch.zeros(len(te), self.max_seq_len, dtype=torch.long)
        for i, d in enumerate(te):
            sentence = d['text'][:self.max_seq_len]
            for j, word in enumerate(sentence):
                if word not in self.word2id:
                    word = '<UNK>'
                test[i, j] = self.word2id[word]
        test_dataset = TensorDataset(test)
        test_sampler=SequentialSampler(test_dataset)
        test_dataloader=DataLoader(test_dataset,sampler=test_sampler,batch_size=self.batch_size)
        return test_dataloader


def main(opt, epochs=1, output_file="data/res.json"):
    processor = DataProcessor()
    weight = processor.get_embedding()
    if torch.cuda.is_available():
        weight=weight.cuda()

    net=LSTMText(opt, weight)
    if torch.cuda.is_available():
        net=net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_dataloader=processor.get_train_data()
    for epoch in trange(epochs, desc="Epoch"):
        eloss=0
        steps=0
        for i,data in enumerate(tqdm(train_dataloader, desc="Iteration")):
            inputs,labels=data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            eloss += loss.item()
            steps+=1
        print("training loss:",eloss/steps)

    # test
    test_dataloader=processor.get_test_data()
    rlist=[]
    net.eval()
    with torch.no_grad():
        for i,inputs in enumerate(tqdm(test_dataloader, desc="Iteration")):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs=net(inputs)
            rlist+=np.argmax(outputs,axis=1).tolist()
    with open(output_file,'w') as f:
        f.write(json.dumps(rlist))


if __name__=='__main__':
    opt = {}
    opt["content_dim"] = 256
    opt["embedding_dim"] = 128
    opt["linear_hidden_size"] = 128
    opt["content_seq_len"] = 500
    opt['dropout']=0.1
    main(opt,epochs=5)


