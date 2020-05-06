# -*- coding: utf-8 -*-


#!pip install transformers

import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
print('TF version',tf.__version__)

MAX_LEN = 96
INPUT_PATH = ""
PATH = f"{INPUT_PATH}tf-roberta/"
tokenizer_one = RobertaTokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt',
    add_prefix_space=True
)
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)
train = pd.read_csv(f'{INPUT_PATH}tweet-sentiment-extraction/train.csv').dropna().reset_index(drop=True)
test = pd.read_csv(f'{INPUT_PATH}tweet-sentiment-extraction/test.csv').dropna().reset_index(drop=True)

num_egs = train.shape[0]
input_ids = np.ones((num_egs,MAX_LEN),dtype='int32')
attention_mask = np.zeros((num_egs,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((num_egs,MAX_LEN),dtype='int32')
start_tokens = np.zeros((num_egs,MAX_LEN),dtype='int32')
end_tokens = np.zeros((num_egs,MAX_LEN),dtype='int32')

num_egs_t = test.shape[0]
input_ids_t = np.ones((num_egs_t, MAX_LEN), dtype='int32')
attention_mask_t = np.zeros((num_egs_t, MAX_LEN), dtype='int32')
token_type_ids_t = np.zeros((num_egs_t, MAX_LEN), dtype='int32')

for k in range(num_egs):
    tweet =  " "+" ".join(train.loc[k,'text'].split())
    selected_text = " "+" ".join(train.loc[k, 'selected_text'].split())
    sentiment = " "+" ".join(train.loc[k,'sentiment'].split())

    # encode using the RobertaTokenizer
    enc_twit = tokenizer_one.encode(tweet) 
    enc_sen = tokenizer_one.encode(sentiment)
    enc_twit_sen = enc_twit + enc_sen

    # enc selcted without first and last
    enc_selected = tokenizer_one.encode(selected_text)[1:-1]

    input_ids[k,:len(enc_twit_sen)] = enc_twit_sen
    attention_mask[k,:len(enc_twit_sen)] = 1
    for ind in (i for i,e in enumerate(enc_twit) if e == enc_selected[0]):
        if enc_twit[ind:ind+len(enc_selected)] == enc_selected:
            start_tokens[k, ind] = 1
            end_tokens[k, ind+(len(enc_selected)-1)] = 1
    # token_type_ids is still all 0's

for k in range(num_egs_t):
    tweet = " "+" ".join(test.loc[k,'text'].split())
    sentiment = " "+" ".join(test.loc[k,'sentiment'].split())
    enc_twit_sentiment = tokenizer_one.encode(tweet) + tokenizer_one.encode(sentiment)
    input_ids_t[k, :len(enc_twit_sentiment)] = enc_twit_sentiment
    attention_mask_t[k, :len(enc_twit_sentiment)] = 1

def build_model():
    ids = tf.keras.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5', config=config)
    x = bert_model(ids, attention_mask=att, token_type_ids=tok)

    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids,att,tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

print(input_ids.shape)

jac = []; VER='v0'; DISPLAY=1 
oof_start = np.zeros((num_egs, MAX_LEN))
oof_end = np.zeros((num_egs,MAX_LEN))
preds_start = np.zeros((num_egs_t,MAX_LEN))
preds_end = np.zeros((num_egs_t,MAX_LEN))

skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=777)

for fold, (idxT, idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):
    K.clear_session()
    model = build_model()

    cp_callback = tf.keras.callbacks.ModelCheckpoint('%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')

    model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
        epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[cp_callback],
        validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
        [start_tokens[idxV,], end_tokens[idxV,]]))

    model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    oof_start[idxV,], oof_end[idxV] = model.predict([input_ids[idxV,], attention_mask[idxV], token_type_ids[idxV,]], verbose=DISPLAY)
    preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
    preds_start = preds[0]/skf.n_splits
    preds_end = preds[1]/skf.n_splits

all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = test.loc[k,'text']
    else:
        text1 = " "+" ".join(test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)

test['selected_text'] = all
test[['textID','selected_text']].to_csv('submission.csv',index=False)
pd.set_option('max_colwidth', 60)
test.sample(25)

