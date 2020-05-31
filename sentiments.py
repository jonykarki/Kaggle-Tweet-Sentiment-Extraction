from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import tqdm
from autocorrect import Speller
import re
import string

punc = set(string.punctuation)
spell = Speller('en')

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def clean_text(text):
    text = reduce_lengthening(text)
    text = text.replace(".", " ")
    clean_list = "".join([c for c in text if c not in punc])
    return spell(clean_list)

sid_obj = SentimentIntensityAnalyzer()

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def polarity_score(text):
    return sid_obj.polarity_scores(text)

def highest_polarity_part(sentence, sentiment="pos"):
    high = {
        "text": "",
        "score": 0
    }
    words_list = sentence.split()
    for i in range(len(words_list)):
        word = words_list[i]
        polarity = polarity_score(word)
        pol_value = polarity["compound"] + polarity[sentiment]
        if pol_value > high["score"]:
            high["text"] = word
            high["score"] = pol_value
        for j in range(i+1, len(words_list)):
            word += " " + words_list[j]
            polarity = polarity_score(word)
            pol_value = polarity["compound"] + polarity[sentiment]
            if pol_value > high["score"]:
                high["text"] = word
                high["score"] = pol_value
    return high

if __name__ == "__main__":
    df = pd.read_csv("positive.csv")
    jaccard_scores = []
    greater_point_five = ""
    for i, j in tqdm.tqdm(df.iterrows(), desc="Calculating", total=df.shape[0]):
        algo = highest_polarity_part(str(j["text"]))["text"]
        if algo == "":
            algo = j["text"]
        score = jaccard(algo, j["selected_text"])
        if score < 0.2:
            greater_point_five += j["text"] + "\n"
            greater_point_five += j["selected_text"] + "\n"
            greater_point_five += algo + "\n"
            greater_point_five += str(score) + "\n"
            greater_point_five += "*********" + "\n"
        jaccard_scores.append(score)
    print(sum(jaccard_scores)/len(jaccard_scores))
    print(len(jaccard_scores))
    print(df.shape)


# sentence = "I hope my mom enjoys her Mother's Day gift"
# selected_text = "enjoys"

# print(highest_polarity_part(sentence))

# while True:
#     text = input("sentence: ")
#     print(highest_polarity_part(text))

# while True:
#     text = input("word: ")
#     score = polarity_score(text)
#     total_score = score["compound"] + score["pos"]
#     print(total_score)