#!/usr/bin/env python
"""Usage:
    cnn_padded.py  --train=FILE --test=FILE [--w2v=FILE]
"""

import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Embedding, Lambda, Merge, Dense, merge, Convolution1D, Activation
from keras.engine import Input
from keras.optimizers import SGD
from keras import backend as K

import os.path

from docopt import docopt
args = docopt(__doc__, version='1.0.0rc2')

def parse_csv(fname,labels_index=None):
    texts1 = []
    texts2 = []
    labels = []
    if labels_index:
        labels_index = labels_index
    else:
        labels_index = {}
    with open(fname, 'rb') as csvfile:
        rdr = csv.reader(csvfile, delimiter='|')
        for row in rdr:
            label = row[0]
            if labels_index.get(label) == None:
                labels_index[row[0]] = len(labels_index)
            labels.append(labels_index.get(label))
            texts1.append(row[1])
            texts2.append(row[2])
    return labels_index, (texts1, texts2, labels)

def prepare_data(train,
                 test,
                 labels_index):
    x1, x2, y = train
    x1_t, x2_t, y_t = test
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x1)
    tokenizer.fit_on_texts(x2)
    seqs1 = tokenizer.texts_to_sequences(x1)
    seqs2 = tokenizer.texts_to_sequences(x2)
    maxlen = len(max(max(seqs1, key=len),
                 max(seqs2, key=len)))
    seqs1_t = tokenizer.texts_to_sequences(x1_t)
    seqs2_t = tokenizer.texts_to_sequences(x2_t)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Max seq length: %s' % maxlen)
    data = [pad_sequences(seqs1, maxlen=maxlen),
            pad_sequences(seqs2, maxlen=maxlen)]
    data_t = [pad_sequences(seqs1_t, maxlen=maxlen),
              pad_sequences(seqs2_t, maxlen=maxlen)]
    if len(labels_index) > 2:
        labels = to_categorical(np.asarray(y))
        labels_t = to_categorical(np.asarray(y_t))
    else:
        labels = np.asarray(y, dtype=np.float32)
        labels_t = np.asarray(y_t, dtype=np.float32)

    indices = np.arange(data[0].shape[0])
    np.random.shuffle(indices)
    data = [data[0][indices], data[1][indices]]
    labels = labels[indices]
    x_train = [data[0], data[1]]
    y_train = labels
    x_test = [data_t[0], data_t[1]]
    y_test = labels_t

    print('Shape of data train tensor:', [x_train[0].shape, x_train[1].shape])
    print('Shape of label train tensor:', y_train.shape)

    print('Shape of data test tensor:', [x_test[0].shape, x_test[1].shape])
    print('Shape of label test tensor:', y_test.shape)

    return maxlen, word_index, (x_train, y_train, x_test, y_test)

def create_embedding_matrix(fname, dim, word_index):
    print("Loading w2v...")
    embeddings_index = {}
    f = open(fname)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def cosine(x):
    axis = lambda a: len(a._keras_shape) - 1
    dot = lambda a, b: K.batch_dot(a, b, axes=axis(a))
    return dot(x[0], x[1]) / K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))

def model_cnn(embedding_matrix,
              word_index, dim, maxlen, clu=300):
    if embedding_matrix:
        W = [embedding_matrix]
    else:
        W = None
    embedding_layer = Embedding(len(word_index) + 1,
                                dim,
                                weights=W,
                                input_length=maxlen,
                                trainable=True)

    x1 = Input(shape=(maxlen,), dtype='int32', name="question_1_input")
    x2 = Input(shape=(maxlen,), dtype='int32', name="question_2_input")
    q1_emb = embedding_layer(x1)
    q2_emb = embedding_layer(x2)
    convolution = Convolution1D(filter_length=3,
                                nb_filter=300,
                                activation='tanh',
                                border_mode='same')
    q1_conv = convolution(q1_emb)
    q2_conv = convolution(q2_emb)
    sumpool = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))
    q1_pool = sumpool(q1_conv)
    q2_pool = sumpool(q2_conv)
    activation = Activation('tanh')
    q1_output = activation(q1_pool)
    q2_output = activation(q2_pool)
    similarity = merge([q1_output, q2_output], mode=cosine, output_shape=(None,1))
    model = Model(input=[x1, x2], output=similarity)
    sgd = SGD(lr=0.005)
    model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['accuracy'])
    print("Model is compiled.")
    return model

if __name__ == '__main__':
    dim = 400
    labels_index, raw_data = parse_csv(args['--train'])
    _, raw_data_t = parse_csv(args['--test'],
                              labels_index=labels_index)
    maxlen, word_index, data = prepare_data(raw_data,
                                            raw_data_t,
                                            labels_index=labels_index)
    if args['--w2v'] != None:
        embedding_matrix = create_embedding_matrix(args['--w2v'],
                                                   dim,
                                                   word_index)
        clu = 300
    else:
        embedding_matrix = None
        clu = 1000
    model = model_cnn(embedding_matrix, word_index, dim, maxlen, clu=clu)
    print('Model summary:')
    model.summary()
    x_train, y_train, x_val, y_val = data
    y_train = np.reshape(y_train,(y_train.shape[0],1,1))
    y_val = np.reshape(y_val,(y_val.shape[0],1,1))
    model.fit(x_train, y_train,
              validation_data=(x_val,y_val),
              nb_epoch=10,
              batch_size=10)

