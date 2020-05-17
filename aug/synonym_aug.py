# trying to replace the words with synonyms
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import pandas as pd 
import numpy as np 
import re

df = pd.read_csv('train_8folds.csv').dropna().reindex()

def decontracted(word):
    # specific
    word = re.sub(r"won\'t", "will not", word)
    word = re.sub(r"can\'t", "can not", word)
    # general
    word = re.sub(r"n\'t", " not", word)
    word = re.sub(r"\'re", " are", word)
    word = re.sub(r"\'s", " is", word)
    word = re.sub(r"\'d", " would", word)
    word = re.sub(r"\'ll", " will", word)
    word = re.sub(r"\'t", " not", word)
    word = re.sub(r"\'ve", " have", word)
    word = re.sub(r"\'m", " am", word)
    return word

pronouns = ['i', "you", "it", "he", "she", "we", "you", "they", "me", "her", "him", "us", "them", "mine", "my", "your", "yours", "their", "theirs", "our", "ours"]

def get_synonym(word):
    if "`" in word:
        word = decontracted(word.replace("`", "'"))
        return word
    if word.lower() in pronouns:
        return word
    word = "".join([i for i in word if i.isalpha()])
    if word in stop_words:
        return word
    for syn in wordnet.synsets(word):
        for name in syn.lemma_names():
            if name.lower() != word.lower():
                return name.replace('_', " ")
                break
    return ""

new_rows = []
for i in range(8):
    df_n = df[df['kfold'].astype(str) == f'{i}']
    for idx, row in df_n.iterrows():
        try:
            index = row['text'].index(row['selected_text'])
        except ValueError:
            index = -1
        if index != -1:
            before_n = [get_synonym(word) for word in str(row['text'])[:index].split()]
            now_n = [get_synonym(word) for word in str(row['text'])[index:index+len(row['selected_text'])].split()]
            after_n = [get_synonym(word) for word in str(row["text"])[index+len(row['selected_text']):].split()]

            text_n = " ".join(before_n + now_n + after_n)
            selected_n = " ".join(now_n)

            new_rows.append([row['textID'], text_n, selected_n, row['sentiment'], row['kfold']])
print(len(new_rows))

new_df = pd.DataFrame(new_rows, columns=["textID", "text", "selected_text", "sentiment", "kfold"])
concat = pd.concat([df, new_df])
concat.to_csv("aug_folds.csv", index=False)