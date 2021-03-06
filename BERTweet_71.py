# BERTweet Highest Score

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

!pip install transformers
!pip install fairseq
!pip install fastBPE

!wget https://public.vinai.io/BERTweet_base_transformers.tar.gz
!tar -xzf *.tar.gz

!export XLA_USE_BF16=1

import os
import torch
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers

import argparse
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 4
INPUT_PATH = "/content/drive/My Drive/Tweet Sentiment Extraction/input/"
OUTPUT_PATH = ""
TRAINING_FILE = f""
ROBERTA_PATH = f""

key = argparse.Namespace(bpe_codes= f"BERTweet_base_transformers/bpe.codes")
bpe = fastBPE(key)

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file("BERTweet_base_transformers/dict.txt")

# TOKENIZER = transformers.RobertaTokenizer(
#     vocab_file =  f'{ROBERTA_PATH}vocab.json',
#     merges_file = f'{ROBERTA_PATH}merges.txt',
#     lowercase = True,
#     add_prefix_space = True
# )

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"
sentiment = "neutral"

# Encode the line using fastBPE & Add prefix <s> and suffix </s> 
subwords = '<s> ' + bpe.encode(line) + ' </s>'
subwords_two = '</s> ' + bpe.encode(sentiment) + ' </s>'
print(subwords)
print(subwords_two)
input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
input_ids_two = vocab.encode_line(subwords_two, append_eos=False, add_if_not_exist=False).long().tolist()
print(input_ids)
print(input_ids_two)
print(vocab[3392])
bpe.decode(" ".join([vocab[i] for i in input_ids]))

class TweetDataset:
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = [' '+' '.join(str(tweet).split()) for tweet in tweets]
        self.sentiments = [' '+' '.join(str(sentiment).split()) for sentiment in sentiments]
        self.selected_texts = [' '+' '.join(str(selected_text).split()) for selected_text in selected_texts]
        self.max_len = MAX_LEN
        
    
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        # enc_tweet_sentiment = self.tokenizer.encode(self.tweets[item]) + self.tokenizer.encode(self.sentiments[item])
        # padding_len = self.max_len - len(enc_tweet_sentiment)
        # input_ids = enc_tweet_sentiment + ([0] * padding_len)
        # attention_mask = ([1] * len(enc_tweet_sentiment)) + ([0] * padding_len)

        # start_index, end_index = 0, 0
        # token_type_ids = [0] * self.max_len

        # enc_selected_text_ids = self.tokenizer.encode(self.selected_texts[item])[1:-1]
        e_tweet = '<s> ' + bpe.encode(self.tweets[item]) + ' </s>'
        enc_tweet = vocab.encode_line(e_tweet, append_eos=False, add_if_not_exist=False).long().tolist()

        e_sentiment = '</s> ' + bpe.encode(self.sentiments[item]) + " " + bpe.encode(self.sentiments[item]) + ' </s>'
        enc_sentiment = vocab.encode_line(e_sentiment, append_eos=False, add_if_not_exist=False).long().tolist()

        enc_tweet_sentiment = enc_tweet + enc_sentiment

        padding_len = self.max_len - len(enc_tweet_sentiment)
        input_ids = enc_tweet_sentiment + ([0] * padding_len)
        attention_mask = ([1] * len(enc_tweet_sentiment)) + ([0] * padding_len)

        start_index, end_index = 0, 0
        token_type_ids = [0] * self.max_len

        e_selected_text_ids = bpe.encode(self.selected_texts[item])
        enc_selected_text_ids = vocab.encode_line(e_selected_text_ids, append_eos=False, add_if_not_exist=False).long().tolist()

        for j in (i for i,e in enumerate(enc_tweet_sentiment) if e == enc_selected_text_ids[0]):
            if enc_tweet_sentiment[j:j+len(enc_selected_text_ids)] == enc_selected_text_ids:
                start_index = j
                end_index = j+(len(enc_selected_text_ids))

        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets_start': torch.tensor(start_index, dtype=torch.long),
            'targets_end': torch.tensor(end_index, dtype=torch.long),
            'orig_tweet': self.tweets[item],
            'orig_selected': self.selected_texts[item],
            'sentiment': self.sentiments[item]
        }

class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            "BERTweet_base_transformers/model.bin",
            config=conf
        )
        self.drop_out = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU()
        self.l0 = nn.Linear(768 * 2, 2)
        self.l1 = nn.Linear(258,1)
        self.l2 = nn.Linear(258,1)

        self.conv1 = nn.Conv1d(MAX_LEN,MAX_LEN,384)
        self.conv2 = nn.Conv1d(MAX_LEN,MAX_LEN,128)

        self.conv3 = nn.Conv1d(MAX_LEN,MAX_LEN,384)
        self.conv4 = nn.Conv1d(MAX_LEN,MAX_LEN,128)

        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out = self.drop_out(out)

        # start_logits = self.conv1(out)
        # start_logits = self.activation(start_logits)
        # start_logits = self.conv2(start_logits)
        # start_logits = self.l1(start_logits)

        # end_logits = self.conv3(out)
        # end_logits = self.activation(end_logits)
        # end_logits = self.conv2(end_logits)
        # end_logits = self.l2(end_logits)

        # return start_logits.squeeze(), end_logits.squeeze()

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

# df_train = pd.read_csv(TRAINING_FILE)

# from pprint import pprint
# train_dataset = TweetDataset(
#     tweets=df_train.text.values,
#     sentiments=df_train.sentiment.values,
#     selected_texts=df_train.selected_text.values
# )

# tweets = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=2,
#     num_workers=1
# )

# model_config = transformers.RobertaConfig.from_pretrained("/content/BERTweet_base_transformers/config.json")
# model_config.output_hidden_states = True
# model = TweetModel(conf=model_config)
# device = xm.xla_device(1)
# model = model.to(device)

# one = None
# two = None
# for tweet in tweets:
#     one = model(
#         tweet['ids'].to(device),
#         tweet['mask'].to(device),
#         tweet['token_type_ids'].to(device)
#     )
#     break

# # pprint(one)
# # print()
# # print(two)

# print(one[-1].shape)
# two = torch.cat((one[-1], one[-2]), dim=-1)
# pprint(two[-1].shape)
# pprint(two[-1][0][0])
# print(one[0].shape)

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss

def train_fn(data_loader, model, optimizer, device, num_batches, scheduler=None):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        scheduler.step()
        tk0.set_postfix(loss=loss.item())

def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            jaccard_scores = []
            for id, tweet in enumerate(orig_tweet):
                a = np.argmax(outputs_start[id])
                b = np.argmax(outputs_end[id])
                if a > b:
                    selected_text = tweet
                else:
                    # enc_tweet = TOKENIZER.encode(tweet)
                    # selected_text = TOKENIZER.decode(enc_tweet[a:b])
                    n_tweet = '<s> ' + bpe.encode(tweet) + ' </s>'
                    nn_tweet = vocab.encode_line(n_tweet, append_eos=False, add_if_not_exist=False).long().tolist()
                    select = nn_tweet[a:b]
                    selected_text = bpe.decode(" ".join([vocab[i] for i in select]))
                    selected_text = selected_text.replace("<s>","").replace("</s>","")
                    if selected_text.strip() == "":
                        selected_text = tweet
                jaccard_scores.append(jaccard(selected_text, orig_selected[id]))

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=loss.item())

    return jaccards.avg

dfx = pd.read_csv(TRAINING_FILE)

def run(fold):
    model_config = transformers.RobertaConfig.from_pretrained(
                "BERTweet_base_transformers/config.json"
            )
    model_config.output_hidden_states = True
    model_config.num_hidden_layers = 14
    MX = TweetModel(conf=model_config)

    device = xm.xla_device(fold+1)
    model = MX.to(device)
    
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = TweetDataset(
        tweets=df_train.text.values,
        sentiments=df_train.sentiment.values,
        selected_texts=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=1
    )

    valid_dataset = TweetDataset(
        tweets=df_valid.text.values,
        sentiments=df_valid.sentiment.values,
        selected_texts=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=1
    )

    num_batches = int(len(df_train)/TRAIN_BATCH_SIZE)
    num_train_steps = num_batches * EPOCHS

    optimizer = transformers.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    best_jac = 0
    # es = EarlyStopping(patience=2, mode="max")
    
    for epoch in range(EPOCHS):
        train_fn(
            train_data_loader, 
            model, 
            optimizer, 
            device,
            num_batches,
            scheduler
        )

        jac = eval_fn(
            valid_data_loader, 
            model, 
            device
        )
        print(f'Epoch={epoch}, Fold={fold}, Jaccard={jac}')
        if jac > best_jac:
            print(f'Saving model {fold}')
            xm.save(model.state_dict(), f"{OUTPUT_PATH}model_{fold}.bin")
            best_jac = jac

Parallel(n_jobs=8, backend="threading")(delayed(run)(i) for i in range(8))

# INFERENCE
df_test = pd.read_csv("test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values

model_config = transformers.RobertaConfig.from_pretrained(
                "/content/BERTweet_base_transformers/config.json"
            )
model_config.output_hidden_states = True

final_output = []
test_dataset = TweetDataset(
        tweets=df_test.text.values,
        sentiments=df_test.sentiment.values,
        selected_texts=df_test.selected_text.values
    )

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)

agreater = []

device = xm.xla_device()

model1 = TweetModel(conf=model_config)
model1.to(device)
model1.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_0.bin"))
model1.eval()

model2 = TweetModel(conf=model_config)
model2.to(device)
model2.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_1.bin"))
model2.eval()

model3 = TweetModel(conf=model_config)
model3.to(device)
model3.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_2.bin"))
model3.eval()

model4 = TweetModel(conf=model_config)
model4.to(device)
model4.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_3.bin"))
model4.eval()

model5 = TweetModel(conf=model_config)
model5.to(device)
model5.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_4.bin"))
model5.eval()

model6 = TweetModel(conf=model_config)
model6.to(device)
model6.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_5.bin"))
model6.eval()

model7 = TweetModel(conf=model_config)
model7.to(device)
model7.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_6.bin"))
model7.eval()

model8 = TweetModel(conf=model_config)
model8.to(device)
model8.load_state_dict(torch.load(f"{OUTPUT_PATH}/model_7.bin"))
model8.eval()

with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start2, outputs_end2 = model2(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start3, outputs_end3 = model3(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start4, outputs_end4 = model4(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start5, outputs_end5 = model5(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start6, outputs_end6 = model6(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start7, outputs_end7 = model7(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start8, outputs_end8 = model8(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start = (outputs_start1 + outputs_start2 + 
                         outputs_start3 + outputs_start4 + 
                         outputs_start5 + outputs_start6 + 
                         outputs_start7 + outputs_start8) / 8

        outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 + 
                       outputs_end5 + outputs_end6 + 
                       outputs_end7 + outputs_end8) / 8
        
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        
        for id, tweet in enumerate(orig_tweet):
            a = np.argmax(outputs_start[id])
            b = np.argmax(outputs_end[id])
            if a > b:
                n_tweet = '<s> ' + bpe.encode(tweet) + ' </s>'
                nn_tweet = vocab.encode_line(n_tweet, append_eos=False, add_if_not_exist=False).long().tolist()
                select = nn_tweet[b:a]
                selected_text = bpe.decode(" ".join([vocab[i] for i in select]))
                selected_text = selected_text.replace("<s>","").replace("</s>","")
                if selected_text.strip() == "":
                    selected_text = tweet
            else:
                n_tweet = '<s> ' + bpe.encode(tweet) + ' </s>'
                nn_tweet = vocab.encode_line(n_tweet, append_eos=False, add_if_not_exist=False).long().tolist()
                select = nn_tweet[a:b]
                selected_text = bpe.decode(" ".join([vocab[i] for i in select]))
                selected_text = selected_text.replace("<s>","").replace("</s>","")
                if selected_text.strip() == "":
                    selected_text = tweet
            final_output.append(selected_text)

agreater

final_output[:10]

final_output[:10]

sample = pd.read_csv("sample_submission.csv")
sample.loc[:, 'selected_text'] = final_output
sample.to_csv("submission.csv", index=False)