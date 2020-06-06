# -*- coding: utf-8 -*-
# WORKING VERSION

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

AUTO = tf.data.experimental.AUTOTUNE

MAX_LEN = 96
EPOCHS = 4
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
MODEL_NAME = 'roberta-base'
LEARNING_RATE = 3e-5

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode_tweets(tweets, sentiments, selected_texts):
    encrypt = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'start_tokens': [],
        'end_tokens': [],
        'selected_texts': []
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
            encrypt['selected_texts'].append(selected_text)
    return encrypt

df = pd.read_csv("/content/train.csv").fillna('')
df_test = pd.read_csv("/content/test.csv").fillna('')
df_test.loc[:, "selected_text"] = df_test.text.values
df.head()


train, valid = train_test_split(df, test_size=0.05, stratify=df.sentiment, random_state=42)
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
test_enc = encode_tweets(
    df_test.text.values,
    df_test.sentiment.values,
    df_test.selected_text.values
)

def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    bert_model = TFAutoModel.from_pretrained(MODEL_NAME)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


with strategy.scope():
    model = build_model()
model.summary()

ids = np.array(train_enc["input_ids"])
att = np.array(train_enc["attention_mask"])
tok = np.array(train_enc["token_type_ids"])
start = np.array(train_enc["start_tokens"])
end = np.array(train_enc["end_tokens"])

ids_v = np.array(valid_enc["input_ids"])
att_v = np.array(valid_enc["attention_mask"])
tok_v = np.array(valid_enc["token_type_ids"])
start_v = np.array(valid_enc["start_tokens"])
end_v = np.array(valid_enc["end_tokens"])

ids_t = np.array(test_enc["input_ids"])
att_t = np.array(test_enc["attention_mask"])
tok_t = np.array(test_enc["token_type_ids"])
start_t = np.array(test_enc["start_tokens"])
end_t = np.array(test_enc["end_tokens"])

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

class JaccardScore(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        preds = model.predict([ids_v, att_v, tok_v], verbose=1)
        jacs = []
        for i in range(ids_v.shape[0]):
            a = np.argmax(preds[0][i])
            b = np.argmax(preds[1][i])
            if b > a:
                text = TOKENIZER.decode(ids_v[i][a:b])
                selected_text_orig = valid_enc["selected_texts"][i]
                jacs.append(jaccard(text, selected_text_orig))
        print(np.mean(jacs))

K.clear_session()
sv = tf.keras.callbacks.ModelCheckpoint(
        'epoch-{epoch:1d}-model.h5', monitor='val_loss', verbose=1, save_best_only=False,
        save_weights_only=True, mode='auto', save_freq='epoch')
train_history = model.fit(
    x=[ids, att, tok],
    y=[start, end],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[JaccardScore(), sv],
    validation_data=(
        [ids_v, att_v, tok_v],
        [start_v, end_v]
    )
)

preds = model.predict([
                       ids_t,
                       att_t,
                       tok_t
], verbose=1)

for i in range(ids_t.shape[0]):
    a = np.argmax(preds[0][i])
    b = np.argmax(preds[1][i])
    if b > a:
        text = TOKENIZER.decode(ids_t[i][a:b])
        print(text)


