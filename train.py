from __future__ import division
import time
import argparse
import math
import random
import sys
import time
import logging
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range
from tqdm import tqdm
from tensorflow.python.ops import variable_scope as vs


class Config(object):
    num_features = sum(1 for line in open('feats.txt') if line != '\n')
    batch_size = 10
    num_epochs = 10
    lr = 1e-4
    max_length = 50
    cell_size = 64

class OurModel():
    def add_placeholders(self):

        # per item in batch, per syllable, features
        self.inputs_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, self.config.num_features), name = 'inputs_placeholder')
        # per item in batch, per syllable, 3 predictions
        self.labels_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, 3), name = 'labels_placeholder')
        # per item in batch, number of syllables
        self.seq_lens_placeholder = tf.placeholder(tf.int64, shape = (self.config.batch_size))


        # inputs_placeholder = tf.placeholder(tf.float32, shape = (config.batch_size, config.max_length, config.num_features), name = 'inputs_placeholder')
        # seq_lens_placeholder = tf.placeholder(tf.int64, shape = (config.batch_size))

    def create_feed_dict(self, inputs_batch, labels_batch, seq_lens_batch):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.seq_lens_placeholder: seq_lens_batch
        } 
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict


    def add_prediction_op(self):
        '''
        cell = tf.contrib.rnn.BasicLSTMCell(self.config.cell_size)
        o, h = tf.nn.dynamic_rnn(cell = cell,
                                 dtype = tf.float32,
                                 sequence_length = self.seq_lens_placeholder,
                                 inputs = self.inputs_placeholder
                                 )
        print self.inputs_placeholder # 10 x 30 x 2
        print self.seq_lens_placeholder # 10
        print self.labels_placeholder # 10 x 30 x 3
        print o # 10 x 30 x 64: batch * length * cell
        print h # tuple of 2x ? x 64?
        assert False

        o2 = tf.reshape(o, (-1, self.config.cell_size))
        W = tf.get_variable('weight', shape = (self.config.cell_size, 3))
        b = tf.get_variable('bias', shape = (self.config.batch_size * self.config.max_length, 3))
        y = tf.reshape(tf.matmul(o2, W) + b, (self.config.batch_size, self.config.max_length, 3))
        '''

        # https://github.com/tensorflow/tensorflow/issues/8191
        # cell = tf.contrib.rnn.BasicLSTMCell(self.config.cell_size, 
        #                                     reuse=tf.get_variable_scope().reuse)
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.cell_size)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.cell_size)
        o, h = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw,
                                               cell_bw = cell_bw,
                                               dtype = tf.float32,
                                               sequence_length = self.seq_lens_placeholder,
                                               inputs = self.inputs_placeholder,
                                               )
        fw_o, bw_o = o
        fw_h, bw_h = h
        o = tf.concat((fw_o, bw_o), 2)

        # print self.inputs_placeholder # 10 x 30 x 2
        # print self.seq_lens_placeholder # 10
        # print self.labels_placeholder # 10 x 30 x 3
        # print fw_o # 10, 30, 64
        # print bw_o # 10, 30, 64
        # print o # 10 x 30 x 128: batch * length * cell x 2
        # assert False 

        o2 = tf.reshape(o, (-1, self.config.cell_size * 2))
        W = tf.get_variable('weight', shape = (self.config.cell_size * 2, 3))
        b = tf.get_variable('bias', shape = (self.config.batch_size * self.config.max_length, 3))
        y = tf.reshape(tf.matmul(o2, W) + b, (self.config.batch_size, self.config.max_length, 3))


        self.pred = y

    def add_loss_op(self):
        # Compute cross entropy for each frame.
        mask = tf.sequence_mask(self.seq_lens_placeholder, self.config.max_length)

        masked_labels = tf.boolean_mask(self.labels_placeholder, mask)
        masked_pred = tf.boolean_mask(self.pred, mask)
        loss = tf.nn.l2_loss(tf.subtract(masked_labels, masked_pred))
        
        self.loss = loss

    def add_training_op(self):
    
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(self.loss)
        
        self.train_op = train_op

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()

    def train_on_batch(self, session, train_inputs_batch, train_labels_batch, train_seq_len_batch):
        feed = self.create_feed_dict(train_inputs_batch, train_labels_batch, train_seq_len_batch)
        # batch_cost, summary = session.run([self.loss, self.merged_summary_op], feed)

        loss, _= session.run([self.loss, self.train_op], feed)

        return loss

    def test_on_batch(self, session, test_inputs_batch, test_labels_batch, test_seq_len_batch):
        feed = self.create_feed_dict(test_inputs_batch, test_labels_batch, test_seq_len_batch)
        loss, = session.run([self.loss], feed)

        return loss


    def __init__(self, config):
        self.config = config
        self.build()

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()



def process_line(line):
    line = line.strip().split(' ')
    line = [float(x) for x in line]
    return line

# force it to be max_feats length, pad teh rest with zeros
def pad(elems, config):
    return np.append(elems, np.zeros((config.max_length, elems.shape[1])), 0)[:config.max_length, :]

def make_batches(config, feats_dir, target_dir):
    inputs = np.array([])
    length = np.array([])

    for feats in os.listdir(feats_dir):
        feats = os.path.join(feats_dir, feats)
        with open(feats) as f:
            elems = np.vstack(process_line(line) for line in f)
        padded_elems = pad(elems, config)

        length = np.append(length, min(elems.shape[0], config.max_length))
        if inputs.shape[0] == 0:
            inputs = np.array([padded_elems])
        else:
            inputs = np.append(inputs, [padded_elems], 0)

    labels = np.array([])
    for f0 in os.listdir(target_dir):
        f0 = os.path.join(target_dir, f0)
        with open(f0) as f:
            elems = np.vstack(process_line(line) for line in f)
        padded_elems = pad(elems, config)

        if labels.shape[0] == 0:
            labels = np.array([padded_elems])
        else:
            labels = np.append(labels, [padded_elems], 0)

    batched_inputs = []
    batched_length = []
    batched_labels = []
    # Subtract so all batches are the same size
    for i in range(0, inputs.shape[0] - config.batch_size, config.batch_size):
        batched_inputs.append(inputs[i:i + config.batch_size])
        batched_length.append(length[i:i + config.batch_size])
        batched_labels.append(labels[i:i + config.batch_size])

    return np.array(batched_inputs), np.array(batched_length), np.array(batched_labels)

def test(feats_dir, target_dir):
    config = Config()

    start = time.time()
    print 'Batching data...'
    batched_inputs, batched_length, batched_labels = make_batches(config, feats_dir, target_dir)

    num_batches = len(batched_inputs)
    num_test = int(0.1 * num_batches)
    test_idxs = np.random.choice(num_batches, num_test)
    train_idxs = list(set(range(num_batches)) - set(test_idxs))
    num_train = len(train_idxs)

    train_inputs = batched_inputs[train_idxs]
    train_labels = batched_labels[train_idxs]
    train_length = batched_length[train_idxs]

    test_inputs = batched_inputs[test_idxs]
    test_labels = batched_labels[test_idxs]
    test_length = batched_length[test_idxs]
    print 'batched in {:3f}'.format(time.time() - start)


    with tf.Graph().as_default():
        model = OurModel(config)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep = 5, #default 5
                               pad_step_number = True, # so that alphasort of models works
                               )


        with tf.Session() as sess:
            start = time.time()
            sess.run(init)
            # train_writer = tf.summary.FileWriter('train', sess.graph)
            if load_from_file is not None:
                saver.restore(sess, load_from_file)

            print 'Model initialized in {:.3f}'.format(time.time() - start)

            global_start = time.time()

            for epoch in range(config.num_epochs):
                train_cost = 0
                test_cost = 0
                start = time.time()

                for batch_idx in tqdm(range(num_train), desc = 'Training'):
                    inputs = train_inputs[batch_idx]
                    labels = train_labels[batch_idx]
                    length = train_length[batch_idx]
                
                    loss = model.train_on_batch(sess, inputs, labels, length)

                    train_cost += loss
                    # train_writer.add_summary(summary, step_ii)
                train_cost = train_cost / num_train / config.batch_size

                for batch_idx in tqdm(range(num_test), desc = 'Testing'):
                    inputs = test_inputs[batch_idx]
                    labels = test_labels[batch_idx]
                    length = test_length[batch_idx]
                
                    loss = model.test_on_batch(sess, inputs, labels, length)

                    test_cost += loss
                test_cost = test_cost / num_test / config.batch_size


                print "Epoch {}/{} | train_cost = {:.3f} | test_cost = {:.3f} | time = {:.3f}".format(epoch + 1, config.num_epochs, train_cost, test_cost, time.time() - start)

                saver.save(sess, save_to_file, global_step = epoch + 1 + last_model_number)


model_name = 'test2'
model_dir = os.path.join('..', 'model')

save_to_file = os.path.join(model_dir, model_name)

models = [file for file in os.listdir(model_dir) if model_name in file and '.index' in file]

# Set True to force it to make a new model
# probably better to just do a new name
new_model = True 
load_from_file = None
last_model_number = 0

if new_model or len(models) == 0:
    print 'New model, no loading'
else:
    last_model = max(models)
    last_model_name = last_model.split('.')[0]
    last_model_number = int(last_model_name.split('-')[-1])
    load_from_file = os.path.join(model_dir, last_model_name)
    print 'Loading from' + load_from_file
    print 'starting saving from checkpoint ' + str(1 + last_model_number)

if __name__ == '__main__':
    test('../ATrampAbroad/feats', '../ATrampAbroad/f0')


















        