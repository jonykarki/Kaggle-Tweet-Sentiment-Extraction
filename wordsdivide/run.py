import csv
import pandas
from word import Word

wordss = []
word_list = []
# word_list.append(Word("Zoo", "Neutral"))
# word_list.append(Word("Apple", "Neutral"))
# word_list.append(Word("Hello", "Neutral"))
# word_list.sort()
# print(word_list)

train = csv.reader(open('train.csv'), delimiter=',')
next(train)
for line in train:
    for wrd in line[2].split():
        if wrd.lower() not in wordss:
            wordss.append(wrd.lower())
            word_list.append(Word(wrd, line[3]))

text_ids = []
selected_texts = []
test = csv.reader(open('test.csv'), delimiter=',')
next(test)
n = 0
for line in test:
    print("Currently at ", n)
    n +=1
    pred_list = []
    for wrd in line[1].split():
        curr = [x for x in word_list if x.word == wrd.lower()]
        if len(curr) == 0:
            if line[2] == 'neutral':
                pred_list.append(wrd.lower())
        else:
            if line[2] == curr[0].sentiment:
                pred_list.append(wrd.lower())
    text_ids.append(line[0])
    selected_texts.append(' '.join(word for word in pred_list))
    
df = pandas.DataFrame(data={"textID": text_ids, "selected_text": selected_texts})
df.to_csv("submission.csv", sep=',', index=False)