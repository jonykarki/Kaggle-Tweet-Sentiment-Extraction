import csv
import pandas
import numpy as np

word_list = {}

train = csv.reader(open('../input/tweet-sentiment-extraction/train.csv'), delimiter=',')
next(train)
for line in train:
    for wrd in line[1].lower().split():
        if wrd in word_list:
            if line[3] == "positive":
                word_list[wrd]["sentiment"] += 1
            elif line[3] == "negative":
                word_list[wrd]["sentiment"] -= 1
            else:
                word_list[wrd]["sentiment"] += 0
        else:
            if line[3] == "positive":
                word_list[wrd] = {"sentiment": 5}
            elif line[3] == "negative":
                word_list[wrd] = {"sentiment": -5}
            else:
                word_list[wrd] = {"sentiment": 0}
    for wrd in line[2].lower().split():
        if wrd in word_list:
            if line[3] == "positive":
                word_list[wrd]["sentiment"] += 5
            elif line[3] == "negative":
                word_list[wrd]["sentiment"] -= 5
            else:
                word_list[wrd]["sentiment"] += 0


text_ids = []
selected_texts = []
test = csv.reader(open('../input/tweet-sentiment-extraction/test.csv'), delimiter=',')
next(test)
for line in test:
    words = []
    all_words = []
    sentiments = []
    for wrd in line[1].lower().split():
        if wrd in word_list:
            words.append({wrd: word_list[wrd]})
            all_words.append(wrd)
            sentiments.append(word_list[wrd]["sentiment"])
        else:
            #improve?
            words.append({wrd: {"sentiment": 0}})
            all_words.append(wrd)
            sentiments.append(0)
            
    selected_words = []
    if line[2] == "positive":
        for wrd in all_words[:np.argmax(sentiments)]:
            if wrd not in selected_words:
                if sentiments[all_words.index(wrd)] > 5:
                    selected_words.append(wrd)
        selected_words += all_words[np.argmax(sentiments):]
    elif line[2] == "negative":
        for wrd in all_words[:np.argmin(sentiments)]:
            if wrd not in selected_words:
                if sentiments[all_words.index(wrd)] < -5:
                    selected_words.append(wrd)
        selected_words += all_words[np.argmin(sentiments):]
    else:
        for sent in sentiments:
            if len(all_words) < 10:
                selected_words = all_words
            elif sent < 5 and sent > -5:
                selected_words += all_words[sentiments.index(sent):]
            else:
                selected_words += all_words[np.argmin(sentiments):]
            break
    text_ids.append(line[0])
    selected_texts.append(' '.join(word for word in selected_words))

df = pandas.DataFrame(data={"textID": text_ids, "selected_text": selected_texts})
df.to_csv("submission.csv", sep=',', index=False)

