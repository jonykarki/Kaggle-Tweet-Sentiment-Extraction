import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

out = {
    "textID": [],
    "text": [],
    "selected_text": [],
    "sentiment": [],
    "validate": [],
}

df_positive = df.loc[df["sentiment"] == "positive"]
a, b, c, d = train_test_split(df_positive, df_positive.sentiment, test_size=0.01)
out["textID"].extend(np.array(a["textID"]))
out["text"].extend(np.array(a["text"]))
out["selected_text"].extend(np.array(a["selected_text"]))
out["sentiment"].extend(np.array(a["sentiment"]))
out["validate"].extend(["no"]*len(a))

out["textID"].extend(np.array(b["textID"]))
out["text"].extend(np.array(b["text"]))
out["selected_text"].extend(np.array(b["selected_text"]))
out["sentiment"].extend(np.array(b["sentiment"]))
out["validate"].extend(["yes"]*len(b))


df_negative = df.loc[df["sentiment"] == "negative"]
a, b, c, d = train_test_split(df_negative, df_negative.sentiment, test_size=0.01)
out["textID"].extend(np.array(a["textID"]))
out["text"].extend(np.array(a["text"]))
out["selected_text"].extend(np.array(a["selected_text"]))
out["sentiment"].extend(np.array(a["sentiment"]))
out["validate"].extend(["no"]*len(a))

out["textID"].extend(np.array(b["textID"]))
out["text"].extend(np.array(b["text"]))
out["selected_text"].extend(np.array(b["selected_text"]))
out["sentiment"].extend(np.array(b["sentiment"]))
out["validate"].extend(["yes"]*len(b))


df_neutral = df.loc[df["sentiment"] == "neutral"]
a, b, c, d = train_test_split(df_neutral, df_neutral.sentiment, test_size=0.01)
out["textID"].extend(np.array(a["textID"]))
out["text"].extend(np.array(a["text"]))
out["selected_text"].extend(np.array(a["selected_text"]))
out["sentiment"].extend(np.array(a["sentiment"]))
out["validate"].extend(["no"]*len(a))

out["textID"].extend(np.array(b["textID"]))
out["text"].extend(np.array(b["text"]))
out["selected_text"].extend(np.array(b["selected_text"]))
out["sentiment"].extend(np.array(b["sentiment"]))
out["validate"].extend(["yes"]*len(b))

dff = pd.DataFrame.from_dict(out)
dff.to_csv("separated.csv", index=False)