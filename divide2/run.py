import csv
import pandas

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
    for wrd in line[1].lower().split():
        if wrd in word_list:
            words.append({wrd: word_list[wrd]})
        else:
            words.append({wrd: {"sentiment": 0}})
    selected_words = []
    if line[2] == "positive":
        for wrd in words:
            for k, v in wrd.items():
                if v["sentiment"] > 5:
                    selected_words.append(k)
    elif line[2] == "negative":
        for wrd in words:
            for k, v in wrd.items():
                if v["sentiment"] < -5:
                    selected_words.append(k)
    else:
        for wrd in words:
            for k, v in wrd.items():
                selected_words.append(k)
    text_ids.append(line[0])
    selected_texts.append(' '.join(word for word in selected_words))

df = pandas.DataFrame(data={"textID": text_ids, "selected_text": selected_texts})
df.to_csv("submission.csv", sep=',', index=False)

