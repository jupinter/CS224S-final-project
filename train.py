import time
import argparse
import math
import random
import sys
import time
import logging
from datetime import datetime
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range
from tqdm import tqdm
from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops.nn import rnn_cell, dynamic_rnn, bidirectional_dynamic_rnn
# tf.nn.dynamic_rnn
# tf.contrib.rnn.GRUCell(


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    num_features = 2
    batch_size = 3
    num_epochs = 1
    # n_epochs = 5
    lr = 1e-4

    max_length = 30

    def __init__(self):
        self.output_path = os.path.join('model','{:%Y%m%d_%H%M%S}'.format(datetime.now()))
        self.model_output = os.path.join(self.output_path, "model.weights")

class OurModel():
    def add_placeholders(self):

        # per item in batch, per syllable, features
        self.input_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, self.config.num_features), name = 'input_placeholder')
        # per item in batch, per syllable, 3 predictions
        self.labels_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, 3), name = 'labels_placeholder')
        # per item in batch, number of syllables
        self.seq_lens_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size))

    def create_feed_dict(self, inputs_batch, targets_batch, seq_lens_batch):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.seq_lens_placeholder: seq_lens_batch
        } 
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        self.feed_dict = feed_dict


    def add_prediction_op(self):
        x = self.input_placeholder
        W = tf.Variable(tf.zeros([self.config.n_features, self.config.n_classes]), name = 'W')
        b = tf.Variable(tf.zeros([self.config.batch_size, self.config.n_classes]), name = 'b')
        prod = tf.matmul(x, W) + b
        pred = softmax(prod)


        self.pred = pred

    def add_loss_op(self, pred):

        loss = cross_entropy_loss(pred, self.labels_placeholder)
        
        self.loss = loss

    def add_training_op(self):
    
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(self.loss)
        
        self.train_op = train_op

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()

    def __init__(self, config):
        self.config = config
        self.build()

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op(self.pred)
        self.add_training_op(self.loss)

def line_to_feats(line):
    line = line.strip().split(' ')
    line = [int(x) for x in line]
    return line


# force it to be max_feats length, pad teh rest with zeros
def pad_feats(elems, config):
    return np.append(elems, np.zeros((config.max_length, config.num_features)), 0)[:config.max_length, :]

def make_batches(config, feats_dir, target_dir):
    inputs = None
    lengths = np.array([])

    for feats in os.listdir(feats_dir):
        feats = os.path.join(feats_dir, feats)
        with open(feats) as f:
            elems = np.vstack(line_to_feats(line) for line in f)
        padded_elems = pad_feats(elems, config)
        lengths = np.append(lengths, elems.shape[0])
        if inputs == None:
            inputs = np.array([padded_elems])
        else:
            inputs = np.append(inputs, [padded_elems], 0)

    labels = None
    for f0 in os.listdir(target_dir):
        pass

    batched_inputs = []
    batched_length = []
    batched_labels = []
    for i in range(0, inputs.shape[0], config.batch_size):
        batched_inputs.append(inputs[i:i + config.batch_size])
        batched_length.append(lengths[i:i + config.batch_size])

    return batched_inputs, batched_length, batched_labels

def test(feats_dir, target_dir):
    config = Config()


    batched_inputs, batched_length, batched_labels = make_batches(config, feats_dir, target_dir)

    num_batches = len(batched_inputs)

    inputs = batched_inputs[0]
    print inputs.shape

    lengths = batched_length[0]
    print lengths.shape


    # with tf.Graph().as_default():
    #     logger.info("Building model...",)
    #     start = time.time()
    #     model = OurModel(config)
    #     logger.info("took %.2f seconds", time.time() - start)
        
    #     init = tf.global_variables_initializer()
    #     saver = tf.train.Saver()

    #     with tf.Session() as sess:
    #         sess.run(init)
    #         saver.restore(sess, model.config.model_output)

    #         global_start = time.time()

    #         for epoch in range(config.num_epochs):
    #             start = time.time()

    #             for batch_idx in range(num_batches):
    #                 inputs = batched_inputs[batch_idx]
    #                 lengths = batched_lengths[batch_idx]
    #                 labels = batched_labels[batch_idx]
    #             losses = model.fit(sess, inputs, labels)

test('../ATrampAbroad/feats', '../ATrampAbroad')


















        