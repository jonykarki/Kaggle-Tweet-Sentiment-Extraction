!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

!export XLA_USE_BF16=1

!pip install transformers

import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from tqdm.autonotebook import tqdm

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

LEARNING_RATE = 4e-5
MAX_LEN = 96
TRAIN_BATCH_SIZE = 52
VALID_BATCH_SIZE = 32
EPOCHS = 3
INPUT_PATH = ""
TRAINING_FILE = f"{INPUT_PATH}tweet-sentiment-extraction/train_8folds.csv"
ROBERTA_PATH = f"{INPUT_PATH}roberta-base/"
TOKENIZER_N = transformers.RobertaTokenizer(
    vocab_file =  f'{ROBERTA_PATH}vocab.json',
    merges_file = f'{ROBERTA_PATH}merges.txt',
    lowercase = True,
    add_prefix_space = True
)

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

class TweetDataset:
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = [' '+' '.join(str(tweet).split()) for tweet in tweets]
        self.sentiments = [' '+' '.join(str(sentiment).split()) for sentiment in sentiments]
        self.selected_texts = [' '+' '.join(str(selected_text).split()) for selected_text in selected_texts]
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        enc_tweet_sentiment = self.tokenizer.encode(self.tweets[item]) + self.tokenizer.encode(self.sentiments[item])
        padding_len = self.max_len - len(enc_tweet_sentiment)
        input_ids = enc_tweet_sentiment + ([0] * padding_len)
        attention_mask = ([1] * len(enc_tweet_sentiment)) + ([0] * padding_len)

        start_index, end_index = 0, 0
        token_type_ids = [0] * self.max_len

        enc_selected_text_ids = self.tokenizer.encode(self.selected_texts[item])[1:-1]
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
        self.roberta = transformers.RobertaModel.from_pretrained(ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

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
                    enc_tweet = TOKENIZER.encode(tweet)
                    selected_text = TOKENIZER.decode(enc_tweet[a:b])
                jaccard_scores.append(jaccard(selected_text, orig_selected[id]))

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=loss.item())

    return jaccards.avg

dfx = pd.read_csv(TRAINING_FILE)

def run(fold):
    model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)
    model_config.output_hidden_states = True
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
            xm.save(model.state_dict(), f"model_{fold}.bin")
            best_jac = jac

Parallel(n_jobs=8, backend="threading")(delayed(run)(i) for i in range(8))