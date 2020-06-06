# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from transformers import *
import tokenizers
print("TF version", tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

MAX_LEN = 100
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MODEL_NAME = 'roberta-base'

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode_tweets(tweets, sentiments, selected_texts):
    encrypt = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'start_tokens': [],
        'end_tokens': []
    }
    for i in range(len(tweets)):
        if str(tweets[i]) != "":
            tweet_plus_sen = "{} {} {}".format(
                " ".join(str(tweets[i]).split()),
                "</s></s>", 
                str(sentiments[i])
                )
            selected_text = " ".join(str(selected_texts[i]).split())
            input_ids = TOKENIZER.encode(tweet_plus_sen)
            selected_text_ids = TOKENIZER.encode(selected_text)[1:-1]
            attention_mask = [1] * len(input_ids)
            start_tokens = [0] * MAX_LEN
            end_tokens = [0] * MAX_LEN

            pad_length = MAX_LEN - len(input_ids)
            attention_mask.extend([0] * pad_length)
            input_ids.extend([0] * pad_length)

            for j in (i for i,e in enumerate(input_ids) if e == selected_text_ids[0]):
                if input_ids[j:j+len(selected_text_ids)] == selected_text_ids:
                    start_tokens[j] = 1
                    # TRY DOING -1
                    end_tokens[j+(len(selected_text_ids))] = 1

            encrypt['input_ids'].append(input_ids[:MAX_LEN])
            encrypt['attention_mask'].append(attention_mask[:MAX_LEN])
            encrypt['token_type_ids'].append([0] * MAX_LEN)
            encrypt['start_tokens'].append(start_tokens)
            encrypt['end_tokens'].append(end_tokens)
    return encrypt

df = pd.read_csv("train.csv").fillna('')
df.head()



train, valid = train_test_split(df, test_size=0.1, stratify=df.sentiment, random_state=42)
train_enc = encode_tweets(
    train.text.values,
    train.sentiment.values,
    train.selected_text.values
)
valid_enc = encode_tweets(
    valid.text.values,
    valid.sentiment.values,
    valid.selected_text.values
)

def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    bert_model = TFAutoModel.from_pretrained(MODEL_NAME)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(256, 2,padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(256, 2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


with strategy.scope():
    model = build_model()
model.summary()

n_steps = train.shape[0] // BATCH_SIZE
train_history = model.fit(
    [train_enc["input_ids"], train_enc["attention_mask"], train_enc["token_type_ids"]],
    [train_enc["start_tokens"], train_enc["end_tokens"]],
    steps_per_epoch=n_steps,
    epochs=EPOCHS,
    validation_data=(
        [valid_enc["input_ids"], valid_enc["attention_mask"], valid_enc["token_type_ids"]],
        [valid_enc["start_tokens"], valid_enc["end_tokens"]]
    )
)

