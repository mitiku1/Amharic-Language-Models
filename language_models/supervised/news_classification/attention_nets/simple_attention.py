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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def scaled_dot_attention(Q, K, V):
    dk = Q.size(-1)
    attention = torch.div(torch.bmm(Q, K.transpose(1, 2)), torch.pow(torch.tensor(dk), 0.5))
    attention = F.softmax(attention, dim=-1)
    output = torch.bmm(attention, V)
    return output
def positional_encoding(model_dim, seq_len):
    p = torch.arange(seq_len).view(1, -1, 1).to(device)
    d = torch.arange(model_dim).view(1, 1, -1).to(device)
    phase = p / 1e4**(d / model_dim)
    
    return torch.where(d.long()%2 == 0, torch.sin(phase), torch.cos(phase))
    
class FeedForward(nn.Module):
    def __init__(self, dim, feature_dim):
        super().__init__()
        self.l1 = nn.Linear(dim, feature_dim)
        self.l2 = nn.Linear(feature_dim, dim)
    def forward(self, inputs):
        out = F.relu(self.l1(inputs))
        out = self.l2(out)
        return out
class Residual(nn.Module):
    def __init__(self, dim, sublayer, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, *inputs):
        out = self.sublayer(*inputs)
        out = self.dropout(out)
        out = self.norm(inputs[-1] + out)
        
        return out
class EncoderLayer(nn.Module):
    def __init__(self, dim = 512, 
                 num_heads =  8, 
                 forward_dim = 2048,
                 dropout = 0.1
                 ):
        super().__init__()
        key_dim = value_dim = dim // num_heads
        self.attention = Residual(
            dim = dim,
            sublayer= MultiHeadAttention(num_heads, dim, key_dim, value_dim ),
            dropout= dropout
        )
        
        self.feedforward = Residual(
            dim = dim, sublayer= FeedForward(dim, forward_dim),dropout= dropout
        )
    def forward(self, inputs):
        src = self.attention(inputs, inputs, inputs)
        return self.feedforward(src)
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_layers=6,
                 dim = 512, 
                 num_heads =  8, 
                 forward_dim = 2048,
                 dropout = 0.1
                 ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(dim = dim, num_heads=num_heads, forward_dim=forward_dim, dropout=dropout) for _ in range(num_layers)
        ])
    def forward(self, inputs):
        seq_len, dim = inputs.size(1), inputs.size(2)
        outputs = inputs + positional_encoding(dim, seq_len)
        for layer in self.layers:
            outputs  = layer(outputs)
        return outputs

class DecoderLayer(nn.Module):
    def __init__(self, 
                 dim = 512, 
                 num_heads =  8, 
                 forward_dim = 2048,
                 dropout = 0.1
                 ):
        super().__init__()
        key_dim = value_dim = dim // num_heads
        self.attention_1 = Residual(
            dim = dim,
            sublayer= MultiHeadAttention(num_heads, dim, key_dim, value_dim ),
            dropout= dropout
        )
        
        self.attention_2 = Residual(
            dim = dim,
            sublayer= MultiHeadAttention(num_heads, dim, key_dim, value_dim ),
            dropout= dropout
        )
        
        self.feedforward = Residual(
            dim = dim, sublayer= FeedForward(dim, forward_dim),dropout= dropout
        )
    def forward(self, targets, memory):
        out = self.attention_1(targets, targets, targets)
        out = self.attention_2(out, out, memory)
        return self.feedforward(out)
    

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 num_layers=6,
                 dim = 512, 
                 num_heads =  8, 
                 forward_dim = 2048,
                 dropout = 0.1
                 ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(dim = dim, num_heads=num_heads, forward_dim=forward_dim, dropout=dropout) for _ in range(num_layers)
        ])
        
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, targets, memory):
        seq_len, dim = targets.size(1), targets.size(2)
        
        outputs = targets + positional_encoding(dim, seq_len)
        for layer in self.layers:
            outputs  = layer(outputs, memory)
        outputs = self.linear(outputs)
        return torch.softmax(outputs, dim=-1)
    
class Transformer(nn.Module):
    def  __init__(self,
                  num_encode_layers = 6, 
                  num_decoder_layers = 6,
                  dim = 512, 
                  num_heads = 8,
                  forward_dim = 2048,
                  dropout = 0.1,
                  activation = nn.ReLU(),
                  vocab_size = 10_000,
                 
                  ):
        super().__init__()  
        self.input_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = TransformerEncoder(
            num_layers=num_encode_layers,
            dim = dim,
            num_heads = num_heads,
            forward_dim= forward_dim,
            dropout = dropout
        )
        
        
    def forward(self, inputs):
        x = self.input_embedding(inputs)
        return self.encoder(x)
        
class AttentionHead(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, key_dim)
        self.k = nn.Linear(in_dim, key_dim)
        self.v = nn.Linear(in_dim, value_dim)
    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        return scaled_dot_attention(q, k, v)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, in_dim, key_dim, value_dim):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            AttentionHead(in_dim, key_dim, value_dim) for i in range(num_heads)
        ])
        self.linear = nn.Linear(value_dim * num_heads, in_dim)
        
        
    def forward(self, query, key, value):
        out = [attention(query, key, value) for attention in self.attention_heads]
        out = torch.cat(out, dim=-1)
        out = self.linear(out)
        return out
        
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
        text_encoded = torch.tensor(self.tokenizer.encode(text).input_ids)
        label = self.labels[idx]
        
        return text_encoded, label
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.transformer = Transformer()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = self.transformer(inputs)
        x = self.pool(x).squeeze(-1)
        
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1)
        x = self.fc2(x)
        
        return x
        
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = 0
    corrects = 0
    total = 0
    for inputs, labels in tqdm.auto.tqdm(loader):
        inputs = inputs.to(device).long()
        labels = labels.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        losses += loss.item() * labels.size(0)
        _, preds = outputs.max(dim=-1)
        
        corrects += (preds == labels).sum()
        
        total += labels.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses/total, corrects / total

def evaluate_model(model, loader, criterion):
    model.eval()
    losses = 0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm.auto.tqdm(loader):
            inputs = inputs.to(device).long()
            labels = labels.to(device)
            outputs = model(inputs)
            
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
    
    
    trainloader = DataLoader(trainset, batch_size = 64, shuffle=True)
    validloader = DataLoader(validset, batch_size = 64, shuffle=False)
    
    model = Classifier(len(trainset.label_mapping))
    
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_history, valid_history = train(model, trainloader, validloader, optimizer, criterion, 100)
    
if __name__ == '__main__':
    main()
    