
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter
import tqdm
from collections import Counter
import numpy as np
import random
import math
import os
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from amtokenizers import AmTokenizer
import IPython


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,text,tokenizer, c = 3, k = 100):
        ''' text: a list of words, all text from the training dataset
            tokenizer: AmTokenizer instance to encode and decode text
        '''
        super(WordEmbeddingDataset,self).__init__()
        # Replace all words in the text article with the number in the dictionary, get() function, set the default value
        self.tokenizer = tokenizer
        self.text_encoded = [word_id for t in tqdm.tqdm(text) for word_id in self.tokenizer.encode(t).input_ids ]
        word_counter = Counter()
        print("Generating word counts")
        word_counter.update(self.text_encoded)
        
        word_counts = np.array([count for count in word_counter.values()],dtype=np.float32)
        word_freqs = word_counts/np.sum(word_counts)
        word_freqs = word_freqs ** (3./4.)
        word_freqs = word_freqs / np.sum(word_freqs)

        
    
        # Convert variables of numpy type to tensor type
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_freqs = torch.Tensor(word_freqs)
        self.k = k
        self.c = c
        
        
    def __len__(self):
        ''' returns the length of the entire data set (all words)
        '''
        return len(self.text_encoded)
    def __getitem__(self,idx):
        ''' This function returns the following data for training
                         -Headword
                         -(Positive) words near this word
                         -K randomly sampled words as negative sample
        '''
        #Find the position of the central word in the text eq. love is 4.
        center_word = self.text_encoded[idx]
        # Find the positive sample near the central word in the text, where C = 3. There are 2*C positive samples
        pos_indices = list(range(idx-self.c,idx)) + list(range(idx+1,idx+self.c+1))
        # If the central word is at the beginning or end, then we turn the negative number to the end or beginning of text. eq -1%10 = 9
        pos_indices = [ i%len(self.text_encoded) for i in pos_indices]
        #Find the number of the central word corresponding to the positive word in the dictionary
        pos_words = self.text_encoded[pos_indices]
        #Find the number of the neutral word corresponding to the negative word in the dictionary
        neg_words = torch.multinomial(self.word_freqs, self.k * pos_words.shape[0], True)
        #Return the number of the central word, postive samples and negative samples
        return center_word,pos_words,neg_words
    
# Define PyTorch model
class EmbeddingModel(nn.Module):
    def __init__(self,vocab_size,embed_size):
        ''' Initialize output and output embedding
        '''
        super(EmbeddingModel,self).__init__()
        #Dictionary size 30000
        self.vocab_size = vocab_size
        #Word dimensions are generally 50, 100, 300 dimensions
        self.embed_size = embed_size
        initrange = 0.5/self.embed_size
        #Initialize a matrix, self.vocab_size * self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size,self.embed_size,sparse=False)
        # Initialize the weight in the matrix and set a random value between -0.5 / self.embed_size to 0.5 / self.embed_size
        self.out_embed.weight.data.uniform_(-initrange,initrange)
        #Initialize a matrix, self.vocab_size * self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size,self.embed_size,sparse=False)
        # Initialize the weights in the matrix and set a random value between -0.5 / self.embed_size and 0.5 / self.embed_size
        self.in_embed.weight.data.uniform_(-initrange,initrange)
    
    def forward(self,input_labels,pos_labels,neg_labels):
        '''
                         input_labels: Headword, [batch_size,]
                         pos_labels: words that have appeared in the context window around the central word [batch_size, (window_size * 2)]
                         neg_labelss: words that have not appeared around the head word, obtained from negative sampling [batch_size, (window_size * 2 * K)]
            return: loss, [batch_size]
        '''
        batch_size = input_labels.size(0)
        #The vector obtained by the # embedding method is only randomly initialized and does not represent any meaning, and there will be no training effects such as word2vec.
        # But you can use this method to assign values ​​first and then learn.
        # [batch_szie,embed_size]
        input_embedding = self.in_embed(input_labels)
        # [batch,(2*C),embed_size]
        pos_embedding = self.out_embed(pos_labels)
        # [batch, (2*C * K),embed_size]
        neg_embedding = self.out_embed(neg_labels)
        
        # [batch_size,(2*C)] torch,bmm() multiplication, unsqueeze() increases the dimension, squeeze() reduces the dimension
        log_pos = torch.bmm(pos_embedding,input_embedding.unsqueeze(2)).squeeze()
        # [batch,(2*C*K)]
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()
        
        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)
        
        loss = log_pos + log_neg
        return -loss
    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()

def seed_all(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    # if you are suing GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
def main():
    
    seed_all(1234)
    LOG_FILE = "log.txt"
    vocab_size = 10_000
    EMBEDDING_SIZE = 128
    data_path = "An-Amharic-News-Text-classification-Dataset/data/Amharic News Dataset.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["article", "category"])
    
    tokenizer =  AmTokenizer(vocab_size = vocab_size, min_frequence= 5 , tokenizer_name="word_piece")
    
    IPython.embed()
    os._exit(0)
    
    
    dataset = WordEmbeddingDataset(df["article"].values, tokenizer)
    dataloader = tud.DataLoader(dataset,batch_size=1024,shuffle=True)
    
    model = EmbeddingModel(vocab_size,EMBEDDING_SIZE)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.2)
    
    for e in range(2):
        t = 0
        for i,(input_labels,pos_labels,neg_labels) in tqdm.auto.tqdm(enumerate(dataloader), total=len(dataloader)):
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()
            
            optimizer.zero_grad()
            loss = model(input_labels,pos_labels,neg_labels).mean()
            loss.backward()
            optimizer.step()
            
            if i % 1000 == 0:
                with open(LOG_FILE,"a") as fout:
                    fout.write("epoch:{},iter:{},loss:{}\n".format(e,i,loss.item()))
                    print("epoch:{},iter:{},loss:{}".format(e,i,loss.item()))
            
           
        embedding_weights = model.input_embeddings()
        np.save("embedding--{}".format(EMBEDDING_SIZE),embedding_weights)
        torch.save(model.state_dict(),"embedding-{}.th".format(EMBEDDING_SIZE))
    # IPython.embed()
    
    
    

if __name__ == '__main__':
    main()
    