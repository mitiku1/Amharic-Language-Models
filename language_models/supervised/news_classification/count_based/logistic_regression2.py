import torch
import pandas as pd 
import numpy as np
import IPython
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from amtokenizers import AmTokenizer
from torch import nn
import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NewsClassificationDatataset(Dataset):
    def __init__(self, df, tokenizer, vocab_size, label_mapping = None):
        super().__init__()
        # self.texts = [tokenizer.encode(text).input_ids for text in df.article.values]
        self.vocab_size = vocab_size
        self.texts = df.article.values
        labels = df.category.values
        
        self.tokenizer = tokenizer
        if label_mapping is None:
            classes = list(set(labels))
            classes.sort()
            self.label_mapping = {classes[i]:i for i in range(len(classes))}
        else:
            self.label_mapping = label_mapping
        self.labels = [self.label_mapping[labels[i]] for i in range(len(labels))]
    
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.texts[idx]
        text_encoded = self.tokenizer.encode(text).input_ids
        label = self.labels[idx]
        output = np.zeros(self.vocab_size)
        for token_id in text_encoded:
            output[token_id]+=1
        return output, label
        
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = 0
    corrects = 0
    total = 0
    for inputs, labels in tqdm.tqdm(loader):
        inputs = inputs.to(device).float()
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
        for inputs, labels in tqdm.tqdm(loader):
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            losses += loss.item() * labels.size(0)
            _, preds = outputs.max(dim=-1)
            
            corrects += (preds == labels).sum()
            
            total += labels.size(0)
        
        
    return losses/total, corrects / total
def main():
    data_path = "An-Amharic-News-Text-classification-Dataset/data/Amharic News Dataset.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["article", "category"])
    vocab_size = 10_000
    EPOCHS = 10
    
    tokenizer  = AmTokenizer(vocab_size, 5 , "byte_bpe", max_length=128)
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 1234)
    
    trainset = NewsClassificationDatataset(train_df, tokenizer, vocab_size)
    validset = NewsClassificationDatataset(valid_df, tokenizer, vocab_size, label_mapping=trainset.label_mapping)
    
    trainloader = DataLoader(trainset, batch_size = 64, shuffle=True)
    validloader = DataLoader(validset, batch_size = 64, shuffle=False)
    
    model = nn.Sequential(
        nn.Linear(vocab_size, len(trainset))
    )
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for i in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion)
        val_loss, val_acc = evaluate_model(model, validloader, criterion)
        print("Epoch: {}/{} train-loss:{} train-acc:{} val-loss:{} valid-acc:{}".format(i+1, EPOCHS, train_loss, train_acc, val_loss, val_acc))
    

if __name__ == '__main__':
    main()
    