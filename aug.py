import pandas as pd
import numpy as np
from random import randrange

df = pd.read_csv('train.csv').dropna().reindex()
id = 'cb774db0d1'
final_array = []
for row in np.array(df):
    tweet = str(row[1])
    selected = str(row[2])
    sentiment = str(row[3])

    for i in range(2):
        roww = df.iloc[randrange(df.shape[0]),]
        if roww['sentiment'] == sentiment:
            idx = str(roww['text']).find(roww['selected_text'])
            text1 = str(roww['text'])[:idx+len(roww['selected_text'])]
            idx2 = str(tweet).find(str(selected))
            text2 = str(text1) + " " + str(selected) + " " + str(tweet[idx2+len(selected):])
            slct = roww['selected_text'] + " " + selected

            if len(text2.split() + slct.split()) < 55:
                final_array.append([id, text2, slct, sentiment])
        
df1 = pd.DataFrame(final_array, columns=["textID", "text", "selected_text", "sentiment"])
concat = pd.concat([df, df1])
concat.to_csv("out.csv", index=False)
            