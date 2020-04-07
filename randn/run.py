import csv
import random
import pandas

text_ids = []
selected_texts = []
random.seed(10)

file = csv.reader(open('test.csv'), delimiter=',')
next(file)
for line in file:
    words = line[1].split()
    rand_words = random.sample(words, int(0.5*len(words)))
    selected = []
    for word in words:
        if word in rand_words:
            selected.append(word)
    selected_texts.append(' '.join(word for word in selected))
    text_ids.append(line[0])

df = pandas.DataFrame(data={"textID": text_ids, "selected_text": selected_texts})
df.to_csv("submission_01.csv", sep=',', index=False)