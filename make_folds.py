import pandas as pd
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=8)
data = pd.read_csv("train.csv")

new_df = {
    "textID": [],
    "text": [],
    "selected_text": [],
    "sentiment": [],
    "kfold": []
}
for a, (i, j) in enumerate(skf.split(data, data.sentiment)):
    new_df["textID"].extend(data.loc[j].textID)
    new_df["text"].extend(data.loc[j].text)
    new_df["selected_text"].extend(data.loc[j].selected_text)
    new_df["sentiment"].extend(data.loc[j].sentiment)
    new_df["kfold"].extend([a]*len(j))

newdf = pd.DataFrame.from_dict(new_df)
# print(newdf.groupby('kfold').count())   
newdf.to_csv("folds_train.csv", index=False) 