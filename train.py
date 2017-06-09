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
from time import gmtime, strftime

# DISABLE TQDM, since it's fast enough to not need it at the moment
def tqdm(l, desc = '', total = 42, *args):
    return l

def process_line(line):
    line = line.strip().split(' ')
    line = [float(x) for x in line]
    return line

feats_dirs = ['../ATrampAbroad/feats_final']
f0_files = ['../ATrampAbroad/pitches.txt']
num_feats = 189

class Config(object):
    # num_features = num_numeric_features
    num_features = num_feats
    batch_size = 10
    num_epochs = 10
    lr = 1.
    lr_decay = .8
    max_length = 50
    cell_size = 128
    regularization = 0

class OurModel():
    def add_placeholders(self):
        # per item in batch, per syllable, features
        self.inputs_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, self.config.num_features), name = 'inputs_placeholder')
        # per item in batch, per syllable, 3 predictions
        self.labels_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, 3), name = 'labels_placeholder')
        # per item in batch, number of syllables
        self.seq_lens_placeholder = tf.placeholder(tf.int64, shape = (self.config.batch_size), name = 'seq_lens_placeholder')
        self.masks_placeholder = tf.placeholder(tf.bool, shape = (self.config.batch_size, self.config.max_length, 3), name = 'masks_placeholder')

    def create_feed_dict(self, inputs_batch, labels_batch, seq_lens_batch, masks_batch):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.seq_lens_placeholder: seq_lens_batch,
            self.masks_placeholder: masks_batch
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
        masked_labels = tf.boolean_mask(self.labels_placeholder, self.masks_placeholder, name = 'masked_labels')
        masked_pred = tf.boolean_mask(self.pred, self.masks_placeholder, name = 'masked_pred')
        loss = tf.nn.l2_loss(tf.subtract(masked_labels, masked_pred))

        # All non-bias trainable variables
        # l2_cost = sum(tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name)
        
        params = tf.trainable_variables()
        # grads = tf.gradients(self.loss, params)
        # self.global_norm = tf.global_norm(grads)
        self.param_norm = sum(tf.nn.l2_loss(param) for param in params if len(param.get_shape()) >= 2)
        
        
        self.loss = loss + self.config.regularization * self.param_norm
        self.base_loss = loss
        tf.summary.scalar("loss", self.loss)
        # self.l2_cost = l2_cost

    def add_training_op(self):
        global_step = tf.Variable(0, trainable=False, name = 'epoch')
        self.global_step_increment = tf.assign_add(global_step, 1, name = 'next_epoch')
        lr = tf.train.exponential_decay(learning_rate = self.config.lr, 
                                        global_step = global_step, 
                                        decay_steps = 1,
                                        decay_rate = self.config.lr_decay)

        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(self.loss)
        
        self.train_op = train_op

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()

    def train_on_batch(self, sess, train_inputs_batch, train_labels_batch, train_seq_len_batch, train_masks_batch):
        feed = self.create_feed_dict(train_inputs_batch, train_labels_batch, train_seq_len_batch, train_masks_batch)
        # batch_cost, summary = sess.run([self.loss, self.merged_summary_op], feed)

        loss, _, summary, param_norm, base_loss = sess.run([self.loss, self.train_op, self.merged_summary_op, self.param_norm, self.base_loss], feed)
        return loss, base_loss, param_norm, summary

    def increment_epoch(self, sess):
        sess.run([self.global_step_increment], {})

    def test_on_batch(self, sess, test_inputs_batch, test_labels_batch, test_seq_len_batch, test_masks_batch):
        feed = self.create_feed_dict(test_inputs_batch, test_labels_batch, test_seq_len_batch, test_masks_batch)
        loss, base_loss, param_norm = sess.run([self.loss, self.base_loss, self.param_norm], feed)

        return loss, base_loss, param_norm


    def __init__(self, config):
        self.config = config
        self.build()

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_summary_op()
        self.add_training_op()


# force it to be max_feats length, pad the rest with zeros
def pad(elems, config):
    return np.append(elems, np.zeros((config.max_length, elems.shape[1])), 0)[:config.max_length, :]

# because it hates me
def stack_on(stack, element):
    if stack.shape[0] == 0:
        return element
    else:
        return np.append(stack, element, 0)

def batch_feats(config, feats_dirs):
    inputs = np.array([])
    length = np.array([])
    final_feats = []
    for feats_dir in feats_dirs:
        num_files = len(os.listdir(feats_dir))
        set_inputs = np.zeros((num_files, config.max_length, num_feats))
        set_length = np.zeros(num_files)
        for idx in tqdm(range(num_files), desc = 'Batching feats'):
            feats = os.path.join(feats_dir, os.listdir(feats_dir)[idx])
            with open(feats) as f:
                elems = np.vstack(process_line(line) for line in f)
            padded_elems = pad(elems, config)
            set_inputs[idx] = padded_elems
            set_length[idx] = min(elems.shape[0], config.max_length)
        inputs = stack_on(inputs, set_inputs)
        length = stack_on(length, set_length)
        final_feats += [feats.split('/')[-1].split('.')[0]]
    batched_inputs = []
    batched_length = []
    for i in range(0, inputs.shape[0] - config.batch_size, config.batch_size):
        batched_inputs.append(inputs[i:i + config.batch_size])
        batched_length.append(length[i:i + config.batch_size])
    return np.array(batched_inputs), np.array(batched_length), final_feats

def batch_f0(config, f0_files, final_feats):
    labels = np.array([])
    masks = np.array([])
    for idx, f0_file in enumerate(f0_files):
        with open(f0_file) as f:
            num_lines = sum(1 for line in f) - 1
        set_labels = np.zeros((num_lines, config.max_length, 3))
        set_masks = np.zeros((num_lines, config.max_length, 3))
        with open(f0_file) as f:
            f.next() # skip first line
            curr_file = ''
            curr_file_number = 0
            for line in tqdm(f, total = num_lines, desc = 'Batching f0s'):
                line = line.strip().split('\t')
                new_file = line[0][-11:]
                nums = []
                elem_mask = []
                for x in line[5:8]:
                    try:
                        nums.append(float(x))
                        elem_mask.append(1)
                    except ValueError: # undefined
                        nums.append(0)
                        elem_mask.append(0)
                if curr_file != new_file:
                    if curr_file == final_feats[idx]:
                        break
                    if curr_file != '':
                        padded_elems = pad(elems, config)
                        padded_mask = pad(mask, config)
                        set_labels[curr_file_number] = padded_elems
                        set_masks[curr_file_number] = padded_mask
                        curr_file_number += 1
                    curr_file = new_file
                    elems = np.array([nums])
                    mask = np.array([elem_mask])
                else:
                    elems = np.append(elems, [nums], 0)
                    mask = np.append(mask, [elem_mask], 0)
        labels = stack_on(labels, set_labels)
        masks = stack_on(masks, set_masks)
    batched_labels = []
    batched_masks = []
    for i in range(0, labels.shape[0] - config.batch_size, config.batch_size):
        batched_labels.append(labels[i:i + config.batch_size])
        batched_masks.append(masks[i:i + config.batch_size])
    return np.array(batched_labels), np.array(batched_masks)

def test():
    config = Config()

    global_start = time.time()
    print 'Batching data...'
    batched_inputs, batched_length, final_feats = batch_feats(config, feats_dirs)
    # print batched_inputs.shape # 375 10 80 189
    # print batched_length.shape # 375 10
    # print 'final feats:', final_feats
    batched_labels, batched_masks = batch_f0(config, f0_files, final_feats)
    # print batched_labels.shape # 375 10 80 3
    # print batched_masks.shape # 375 10 80 3

    num_batches = len(batched_inputs)
    num_dev = int(0.1 * num_batches)
    dev_idxs = np.random.choice(num_batches, num_dev)
    train_idxs = list(set(range(num_batches)) - set(dev_idxs))
    num_train = len(train_idxs)

    train_inputs = batched_inputs[train_idxs]
    train_labels = batched_labels[train_idxs]
    train_length = batched_length[train_idxs]
    train_masks = batched_masks[train_idxs]

    dev_inputs = batched_inputs[dev_idxs]
    dev_labels = batched_labels[dev_idxs]
    dev_length = batched_length[dev_idxs]
    dev_masks = batched_masks[dev_idxs]
    print 'Batched in {:3f}'.format(time.time() - global_start)

    train_scale = num_train * config.batch_size
    dev_scale = num_dev * config.batch_size

    with tf.Graph().as_default():
        model = OurModel(config)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep = 1, #default 5
                               pad_step_number = True, # so that alphasort of models works
                               )


        with tf.Session() as sess:
            start = time.time()
            sess.run(init)
            # train_writer = tf.summary.FileWriter('train', sess.graph)
            if load_from_file is not None:
                saver.restore(sess, load_from_file)

            print 'Model initialized in {:.3f}'.format(time.time() - start)
            train_writer = tf.summary.FileWriter(logs_path + '/train', sess.graph)

            global_start = time.time()
            step = 0
            for epoch in range(config.num_epochs):
                train_base_loss = 0
                train_cost = 0
                train_param = 0
                start = time.time()

                for batch_idx in tqdm(range(num_train), desc = 'Train'):
                    inputs = train_inputs[batch_idx]
                    labels = train_labels[batch_idx]
                    length = train_length[batch_idx]
                    masks = train_masks[batch_idx]
                    loss, base_loss, param, summary = model.train_on_batch(sess, inputs, labels, length, masks)
                    train_cost += loss
                    train_param += param
                    train_base_loss += base_loss
                    train_writer.add_summary(summary, step)
                    step += 1
                train_cost /= train_scale
                train_param /= train_scale
                train_base_loss /= train_scale

                dev_cost = 0
                dev_param = 0
                dev_base_loss = 0
                for batch_idx in tqdm(range(num_dev), desc = 'Dev'):
                    inputs = dev_inputs[batch_idx]
                    labels = dev_labels[batch_idx]
                    length = dev_length[batch_idx]
                    masks = dev_masks[batch_idx]
                    loss, base_loss, param = model.test_on_batch(sess, inputs, labels, length, masks)
                    dev_cost += loss
                    dev_param += param
                    dev_base_loss += base_loss
                dev_cost /= dev_scale
                dev_param /= dev_scale
                dev_base_loss /= dev_scale

                print "Epoch {}/{} | train_cost = {:.3f} ({:.3f} w/o reg)| dev_cost = {:.3f} ({:.3f} w/o reg)| train_param = {:.3f} | dev_param = {:.3f} | time = {:.3f}".format(epoch + 1, config.num_epochs, train_cost, train_base_loss, dev_cost, dev_base_loss, train_param, dev_param, time.time() - start)

                model.increment_epoch(sess)

                saver.save(sess, logs_path, global_step = epoch + 1 + last_model_number)
    # print 'total duration: {:.3f}'.format(time.time() - global_start)

model_name = 'many_features'
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

logs_path = os.path.join('..', 'tensorboard', strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

if __name__ == '__main__':
    test()


















        