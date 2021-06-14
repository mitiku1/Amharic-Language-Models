import torch
import pandas as pd 
import numpy as np
import IPython
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from amtokenizers import AmTokenizer
from torch.nn import functional as F
from torch import nn
import torch.nn.functional as F
import tqdm
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TransformerModel(nn.Module):
    def __init__(self, num_layers = 6, emb_size = 512, vocab_size = 10_000, nhead=8, dim_feedforward=512, dropout=0.1, num_classes = 6):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward = dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.linear = nn.Linear(128, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, inputs, attention_masks=None, padding_mask = None):
        inputs_emb = self.positional_encoding(self.token_emb(inputs))

        memory = self.encoder(inputs_emb, attention_masks, padding_mask)
        memory = F.relu(self.pool(memory).squeeze(-1))
        
        output = self.linear(memory)
        
        return output
        


        
class NewsClassificationDatataset(Dataset):
    def __init__(self, df, tokenizer, label_mapping = None):
        super().__init__()
        # self.texts = [tokenizer.encode(text).input_ids for text in df.article.values]
        self.texts = df.article.values
        labels = df.category.values
        if label_mapping is None:
            classes = list(set(labels))
            classes.sort()
            self.label_mapping = {classes[i]:i for i in range(len(classes))}
        else:
            self.label_mapping = label_mapping
        self.tokenizer = tokenizer
        
        self.labels = [self.label_mapping[labels[i]] for i in range(len(labels))]
    
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.texts[idx]
        text_encoded = self.tokenizer.encode(text)
        label = self.labels[idx]
        text_encoded["input_ids"] = torch.tensor(text_encoded["input_ids"])
        text_encoded["attention_mask"] = torch.tensor(text_encoded["attention_mask"])
        return text_encoded, label
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = 0
    corrects = 0
    total = 0
    batch_index = 0
    for inputs, labels in tqdm.auto.tqdm(loader):
        
        inputs["input_ids"] = inputs["input_ids"].to(device).long()
        inputs["attention_mask"] = inputs["attention_mask"].to(device).float()
        
        labels = labels.to(device)
        outputs = model(inputs["input_ids"])
        
        loss = criterion(outputs, labels)
        losses += loss.item() * labels.size(0)
        _, preds = outputs.max(dim=-1)
        
        corrects += (preds == labels).sum()
        
        total += labels.size(0)
        if batch_index % 50 == 0:
            print("{}/{} loss: {} acc: {}".format(batch_index, len(loader), losses/total, corrects/total))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_index += 1
    return losses/total, corrects / total

def evaluate_model(model, loader, criterion):
    model.eval()
    losses = 0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm.auto.tqdm(loader):
            inputs["input_ids"] = inputs["input_ids"].to(device).long()
            inputs["attention_mask"] = inputs["attention_mask"].to(device).float()
       
            labels = labels.to(device)
            outputs = model(inputs["input_ids"])
            
            loss = criterion(outputs, labels)
            losses += loss.item() * labels.size(0)
            _, preds = outputs.max(dim=-1)
            
            corrects += (preds == labels).sum()
            
            total += labels.size(0)
        
        
    return losses/total, corrects / total

def train(model, trainloader, valloader, optimizer, criterion, epochs):
    train_history = {"loss":[], "acc":[]}
    val_history = {"loss":[], "acc":[]}
    for i in range(epochs):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion)
        val_loss, val_acc = evaluate_model(model, valloader, criterion)
        print("Epoch: {}/{} train-loss:{:.4f} train-acc:{:.4f} val-loss:{:.4f} valid-acc:{:.4f}".format(i+1, epochs, train_loss, train_acc, val_loss, val_acc))
        train_history["loss"].append(train_loss)
        train_history["acc"].append(train_acc)
        
        val_history["loss"].append(val_loss)
        val_history["acc"].append(val_acc)
    return train_history, val_history

def main():
    data_path = "An-Amharic-News-Text-classification-Dataset/data/Amharic News Dataset.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["article", "category"])
    tokenizer  = AmTokenizer(10000, 5 , "byte_bpe", max_length=128)
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 1234)
    
    trainset = NewsClassificationDatataset(train_df, tokenizer)
    validset = NewsClassificationDatataset(valid_df, tokenizer, label_mapping=trainset.label_mapping)
    
    
    trainloader = DataLoader(trainset, batch_size = 256, shuffle=True)
    validloader = DataLoader(validset, batch_size = 256, shuffle=False)
    
    model = TransformerModel(num_classes= len(trainset.label_mapping))
    IPython.embed()
    # model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)
    # criterion = nn.CrossEntropyLoss()
    
    # train_history, valid_history = train(model, trainloader, validloader, optimizer, criterion, 100)
    
if __name__ == '__main__':
    main()
    