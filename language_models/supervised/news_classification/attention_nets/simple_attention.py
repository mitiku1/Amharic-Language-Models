import torch
import pandas as pd 
import numpy as np
import IPython
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from amtokenizers import AmTokenizer


class NewsClassificationDatataset(Dataset):
    def __init__(self, df, tokenizer):
        super().__init__()
        # self.texts = [tokenizer.encode(text).input_ids for text in df.article.values]
        self.texts = df.article.values
        labels = df.category.values
        classes = list(set(labels))
        classes.sort()
        self.tokenizer = tokenizer
        self.class2index = {classes[i]:i for i in range(len(classes))}
        self.index2class = {value:key for key, value in self.class2index.items()}
        self.labels = [self.class2index[labels[i]] for i in range(len(labels))]
    
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        text = self.texts[idx]
        text_encoded = self.tokenizer.encode(text)
        label = self.labels[idx]
        
        return text_encoded.input_ids, label
        

def main():
    data_path = "An-Amharic-News-Text-classification-Dataset/data/Amharic News Dataset.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["article", "category"])
    tokenizer  = AmTokenizer(10000, 5 , "byte_bpe", max_length=128)
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 1234)
    
    trainset = NewsClassificationDatataset(train_df, tokenizer)
    IPython.embed()

if __name__ == '__main__':
    main()
    