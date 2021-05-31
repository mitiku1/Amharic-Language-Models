import pandas as pd
import os
import IPython
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer



normalization_mapping = {'[ሃኅኃሐሓኻ]': 'ሀ', '[ሑኁዅ]': 'ሁ', '[ኂሒኺ]': 'ሂ', 
                         '[ኌሔዄ]': 'ሄ', '[ሕኅ]': 'ህ', '[ኆሖኾ]': 'ሆ', '[ሠ]': 'ሰ',
                         '[ሡ]': 'ሱ', '[ሢ]': 'ሲ', '[ሣ]': 'ሳ', '[ሤ]': 'ሴ',
                         '[ሥ]': 'ስ', '[ሦ]': 'ሶ', '[ዓኣዐ]': 'አ', '[ዑ]': 'ኡ',
                         '[ዒ]': 'ኢ', '[ዔ]': 'ኤ', '[ዕ]': 'እ', '[ዖ]': 'ኦ',
                         '[ጸ]': 'ፀ', '[ጹ]': 'ፁ', '[ጺ]': 'ፂ', '[ጻ]': 'ፃ', 
                         '[ጼ]': 'ፄ', '[ጽ]': 'ፅ', '[ጾ]': 'ፆ', '(ሉ[ዋአ])': 'ሏ', 
                         '(ሙ[ዋአ])': 'ሟ', '(ቱ[ዋአ])': 'ቷ', '(ሩ[ዋአ])': 'ሯ', 
                         '(ሱ[ዋአ])': 'ሷ', '(ሹ[ዋአ])': 'ሿ', '(ቁ[ዋአ])': 'ቋ',
                         '(ቡ[ዋአ])': 'ቧ', '(ቹ[ዋአ])': 'ቿ', '(ሁ[ዋአ])': 'ኋ', 
                         '(ኑ[ዋአ])': 'ኗ', '(ኙ[ዋአ])': 'ኟ', '(ኩ[ዋአ])': 'ኳ', '(ዙ[ዋአ])': 'ዟ',
                         '(ጉ[ዋአ])': 'ጓ', '(ደ[ዋአ])': 'ዷ', '(ጡ[ዋአ])': 'ጧ', '(ጩ[ዋአ])': 'ጯ', 
                         '(ጹ[ዋአ])': 'ጿ', '(ፉ[ዋአ])': 'ፏ', '[ቊ]': 'ቁ', '[ኵ]': 'ኩ'}


def normalize_char_level_missmatch(input_token):
    output = input_token
    for key, value in normalization_mapping.items():
        output = re.sub(key, value, output)
    return output


    

def main():
    dataset_path = "An-Amharic-News-Text-classification-Dataset/data/Amharic News Dataset.csv"
    df = pd.read_csv(dataset_path)
    
    df = df.dropna(subset=["article", "category"])
    
    df['article'] = df['article'].str.replace('[^\w\s]','')
    df['article'] = df['article'].apply(lambda x: normalize_char_level_missmatch(x))
    
    unique_labels = df["category"].unique().tolist()
    unique_labels.sort()
    class2index = {unique_labels[i]:i for i in range(len(unique_labels))}
    
    df["category"] = df.category.apply(lambda x: class2index[x])
    
    texts,labels = df['article'].values,df['category'].values
    

    
    
    print("Using Counter vectorizer")
    vectorizer = CountVectorizer(analyzer='word',max_features=1000,ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts).toarray()
    
    
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels,test_size=0.2, random_state = 1234)

    print("Training naive bayes")
    
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    
    y_pred = classifier.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy", accuracy)
    
    print("Classification report")
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    
    
    print("Using TF-IDF")
    vectorizer = TfidfVectorizer(analyzer='word',max_features=1000,ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts).toarray()
    
    
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels,test_size=0.2, random_state = 1234)

    print("Training naive bayes")
    
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    
    y_pred = classifier.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy", accuracy)
    
    print("Classification report")
    print(classification_report(y_test, y_pred, target_names=unique_labels))


    
    

if __name__ == '__main__':
    main()
    