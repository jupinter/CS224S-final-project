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
# def tqdm(l, desc = '', total = 42, *args):
#     return l

def process_line(line):
    line = line.strip().split(' ')
    line = [float(x) for x in line]
    return line

feats_dirs = ['../ATrampAbroad/feats_final', '../LifeOnTheMississippi/feats_final', '../TheAdventuresOfTomSawyer/feats_final', '../TheManThatCorruptedHadleyburg/feats_final']
f0_files = ['../ATrampAbroad/pitches.txt', '../LifeOnTheMississippi/pitches.txt', '../TheAdventuresOfTomSawyer/pitches.txt', '../TheManThatCorruptedHadleyburg/pitches.txt']
num_feats = 189

class Config(object):
    num_features = num_feats
    batch_size = 50
    num_epochs = 40
    lr = 1.
    lr_decay = .85
    max_length = 60
    cell_size = 256
    regularization = 1e-4
    dev_percent = 0.2

class OurModel():
    def add_placeholders(self):
        # per item in batch, per syllable, features
        self.inputs_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, self.config.num_features), name = 'inputs_placeholder')
        # per item in batch, per syllable, 3 predictions
        self.labels_placeholder = tf.placeholder(tf.float32, shape = (self.config.batch_size, self.config.max_length, 3), name = 'labels_placeholder')
        # per item in batch, number of syllables
        self.seq_lens_placeholder = tf.placeholder(tf.int64, shape = (self.config.batch_size), name = 'seq_lens_placeholder')
        self.masks_placeholder = tf.placeholder(tf.bool, shape = (self.config.batch_size, self.config.max_length, 3), name = 'masks_placeholder')

    def create_feed_dict(self, inputs_batch, seq_lens_batch, labels_batch = None, masks_batch = None):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.seq_lens_placeholder: seq_lens_batch,
        } 
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if masks_batch is not None:
            feed_dict[self.masks_placeholder] = masks_batch
        return feed_dict


    def add_prediction_op(self):
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
        # fw_h, bw_h = h
        o = tf.concat((fw_o, bw_o), 2)

        o2 = tf.reshape(o, (-1, self.config.cell_size * 2))
        W = tf.get_variable('weight', shape = (self.config.cell_size * 2, 3))
        b = tf.get_variable('bias', shape = (self.config.batch_size * self.config.max_length, 3))
        y = tf.reshape(tf.matmul(o2, W) + b, (self.config.batch_size, self.config.max_length, 3))


        self.pred = y

    def add_loss_op(self):
        masked_labels = tf.boolean_mask(self.labels_placeholder, self.masks_placeholder, name = 'masked_labels')
        masked_pred = tf.boolean_mask(self.pred, self.masks_placeholder, name = 'masked_pred')
        loss = tf.nn.l2_loss(tf.subtract(masked_labels, masked_pred))
        
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        self.global_norm = tf.global_norm(grads, name = 'grad_norm')
        self.param_norm = sum(tf.nn.l2_loss(param) for param in params if len(param.get_shape()) >= 2)
        
        self.loss = loss + self.config.regularization * self.param_norm
        self.base_loss = loss

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar('base loss', self.base_loss)
        tf.summary.scalar('param norm', self.param_norm)
        tf.summary.histogram('gradients', self.global_norm)

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
        feed = self.create_feed_dict(train_inputs_batch, 
                                     train_seq_len_batch, 
                                     train_labels_batch, 
                                     train_masks_batch)

        loss, _, summary, param_norm, base_loss = sess.run([self.loss, self.train_op, self.merged_summary_op, self.param_norm, self.base_loss], feed)
        return loss, base_loss, param_norm, summary

    def increment_epoch(self, sess):
        sess.run([self.global_step_increment], {})

    def dev_on_batch(self, sess, dev_inputs_batch, dev_labels_batch, dev_seq_len_batch, dev_masks_batch):
        feed = self.create_feed_dict(dev_inputs_batch, 
                                     dev_seq_len_batch, 
                                     dev_labels_batch, 
                                     dev_masks_batch)
        loss, base_loss, param_norm = sess.run([self.loss, self.base_loss, self.param_norm], feed)

        return loss, base_loss, param_norm
    def test_on_batch(self, sess, test_inputs_batch, test_labels_batch, test_seq_lens_batch, test_masks_batch):
        feed = self.create_feed_dict(inputs_batch = test_inputs_batch, 
                                     seq_lens_batch = test_seq_lens_batch, 
                                     labels_batch = test_labels_batch, 
                                     masks_batch = test_masks_batch)
        loss, param_norm, pred = sess.run([self.loss, self.param_norm, self.pred], feed)

        return loss, param_norm, pred
    def predict_on_batch(self, sess, inputs_batch, seq_lens_batch):
        feed = self.create_feed_dict(inputs_batch, seq_lens_batch)
        pred = sess.run([self.pred], feed)[0]
        return pred

    def __init__(self, config):
        self.config = config
        self.build()

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_summary_op()
        self.add_training_op()


# force elems to be configs.max_length length, pad the rest with zeros
def pad(elems, config):
    return np.append(elems, np.zeros((config.max_length, elems.shape[1])), 0)[:config.max_length, :]

# because it hates me, and I don't know a better way of doing this
def stack_on(stack, element):
    if stack.shape[0] == 0:
        return element
    else:
        return np.append(stack, element, 0)

def batch_feats(config, feats_dirs):
    inputs = np.array([])
    length = np.array([])
    feats_counts = []
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
        feats_counts += [num_files]
    batched_inputs = []
    batched_length = []
    for i in range(0, inputs.shape[0] - config.batch_size, config.batch_size):
        batched_inputs.append(inputs[i:i + config.batch_size])
        batched_length.append(length[i:i + config.batch_size])
    return np.array(batched_inputs), np.array(batched_length), feats_counts

def batch_f0(config, f0_files, feats_counts):
    labels = np.array([])
    masks = np.array([])
    for idx, f0_file in enumerate(f0_files):
        num_feats = feats_counts[idx]
        with open(f0_file) as f:
            num_lines = sum(1 for line in f) - 1
        set_labels = np.zeros((num_feats, config.max_length, 3))
        set_masks = np.zeros((num_feats, config.max_length, 3))
        with open(f0_file) as f:
            f.next() # skip first line
            curr_file = ''
            curr_file_number = 0
            for file_line in tqdm(f, total = num_lines, desc = 'Batching f0s'):
                line = file_line.strip().split('\t')
                new_file = line[0][-11:] # '' if line is empty
                nums = []
                elem_mask = []
                for x in line[5:8]:
                    try:
                        nums.append(float(x))
                        elem_mask.append(1)
                    except ValueError: # undefined
                        nums.append(0)
                        elem_mask.append(0)
                if curr_file != new_file: # end of reading
                    if curr_file != '': # if current is not empty, add
                        padded_elems = pad(elems, config)
                        padded_mask = pad(mask, config)
                        set_labels[curr_file_number] = padded_elems
                        set_masks[curr_file_number] = padded_mask
                        curr_file_number += 1
                    elems = np.array([nums]) #initialize
                    mask = np.array([elem_mask])
                    curr_file = new_file
                    if curr_file_number == num_feats:
                        break # all done
                else: # just append for current file
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

def evaluate(model_location):
    config = Config()
    global_start = time.time()

    feats_dir = '../ATrampAbroad/feats_final_test'
    f0_file = '../ATrampAbroad/pitches_test.txt'
    predictions_dir = '../ATrampAbroad/predictions'

    batched_inputs, batched_length, feats_counts = batch_feats(config, [feats_dir])
    batched_labels, batched_masks = batch_f0(config, [f0_file], feats_counts)

    num_test = len(batched_inputs)
    test_scale = num_test * config.batch_size
    print 'Batched in {:3f}'.format(time.time() - global_start)
    outfiles = [x.split('.')[0] + '.txt' for x in os.listdir(feats_dir)]
    with tf.Graph().as_default():
        start = time.time()
        model = OurModel(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep = 1, #default 5
                               pad_step_number = True, # so that alphasort of models works
                               )
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_location)
            print 'Model initialized in {:.3f}'.format(time.time() - start)

            test_cost = 0
            test_param = 0
            out_idx = 0
            for batch_idx in tqdm(range(num_test), desc='Testing'):
                inputs = batched_inputs[batch_idx]
                labels = batched_labels[batch_idx]
                length = batched_length[batch_idx]
                masks = batched_masks[batch_idx]
                loss, param, pred = model.test_on_batch(sess, inputs, labels, length, masks)
                test_cost += loss
                test_param += param
                for i, x in enumerate(pred):
                    outfile = os.path.join(predictions_dir, outfiles[out_idx])
                    l = length[i]
                    with open(outfile, 'w') as f:
                        for j, line in enumerate(x):
                            if j >= l:
                                break
                            f.write('{:.8f}\t{:.8f}\t{:.8f}\n'.format(line[0], line[1], line[2]))
                    out_idx += 1
            test_cost /= test_scale
            test_param /= test_scale
        print 'test cost {:.3f} | test param {:.3f}'.format(test_cost, test_param)

def predict(model_location):
    config = Config()
    global_start = time.time()

    feats_dir = '../test/feats_final'
    predictions_dir = '../test/predictions'

    batched_inputs, batched_length, feats_counts = batch_feats(config, [feats_dir])

    num_test = len(batched_inputs)
    test_scale = num_test * config.batch_size
    print 'Batched in {:3f}'.format(time.time() - global_start)
    outfiles = [x.split('.')[0] + '.txt' for x in os.listdir(feats_dir)]
    with tf.Graph().as_default():
        start = time.time()
        model = OurModel(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep = 1, #default 5
                               pad_step_number = True, # so that alphasort of models works
                               )
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_location)
            print 'Model initialized in {:.3f}'.format(time.time() - start)

            out_idx = 0
            for batch_idx in tqdm(range(num_test), desc='Predicting'):
                inputs = batched_inputs[batch_idx]
                length = batched_length[batch_idx]
                pred = model.predict_on_batch(sess, inputs, length)
                for i, x in enumerate(pred):
                    outfile = os.path.join(predictions_dir, outfiles[out_idx])
                    with open(outfile, 'w') as f:
                        l = length[i]
                        for j, line in enumerate(x):
                            if j >= l:
                                break
                            f.write('{:.8f}\t{:.8f}\t{:.8f}\n'.format(line[0], line[1], line[2]))
                    out_idx += 1


def train():
    config = Config()

    global_start = time.time()
    print 'Batching data...'
    batched_inputs, batched_length, feats_counts = batch_feats(config, feats_dirs)
    batched_labels, batched_masks = batch_f0(config, f0_files, feats_counts)

    num_batches = len(batched_inputs)
    num_dev = int(config.dev_percent * num_batches)
    dev_idxs = np.random.choice(num_batches, num_dev, replace = False)
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

    print '{} batches of size {}, {} training, {} dev'.format(num_batches, config.batch_size, num_train, num_dev)

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
            if load_from_file is not None:
                saver.restore(sess, load_from_file)

            print 'Model initialized in {:.3f}'.format(time.time() - start)
            train_writer = tf.summary.FileWriter(logs_path + '/train', sess.graph)

            global_start = time.time()
            step = 0
            for epoch in range(config.num_epochs):
                train_cost = 0
                start = time.time()
                l = list(range(num_train))
                random.shuffle(l)
                for batch_idx in tqdm(l, desc = 'Train'):
                    inputs = train_inputs[batch_idx]
                    labels = train_labels[batch_idx]
                    length = train_length[batch_idx]
                    masks = train_masks[batch_idx]
                    loss, _, _, summary = model.train_on_batch(sess, inputs, labels, length, masks)
                    train_cost += loss
                    train_writer.add_summary(summary, step)
                    step += 1
                train_cost /= train_scale

                dev_cost = 0
                dev_param = 0
                for batch_idx in tqdm(range(num_dev), desc = 'Dev'):
                    inputs = dev_inputs[batch_idx]
                    labels = dev_labels[batch_idx]
                    length = dev_length[batch_idx]
                    masks = dev_masks[batch_idx]
                    loss, _, param = model.dev_on_batch(sess, inputs, labels, length, masks)
                    dev_cost += loss
                    dev_param += param
                dev_cost /= dev_scale
                dev_param /= dev_scale

                print "Epoch {}/{} | train_cost = {:.3f} | dev_cost = {:.3f} | param = {:.3f} | time = {:.3f}".format(epoch + 1, config.num_epochs, train_cost, dev_cost, dev_param, time.time() - start)

                model.increment_epoch(sess)

                saver.save(sess, logs_path, global_step = epoch + 1 + last_model_number)
    # print 'total duration: {:.3f}'.format(time.time() - global_start)

# model_name = 'model'
# model_dir = os.path.join('..', 'model')
# save_to_file = os.path.join(model_dir, model_name)
# models = [file for file in os.listdir(model_dir) if model_name in file and '.index' in file]

# # Set True to force it to make a new model
# # probably better to just do a new name
# new_model = True 
# load_from_file = None
# last_model_number = 0

# if new_model or len(models) == 0:
#     print 'New model, no loading'
# else:
#     last_model = max(models)
#     last_model_name = last_model.split('.')[0]
#     last_model_number = int(last_model_name.split('-')[-1])
#     load_from_file = os.path.join(model_dir, last_model_name)
#     print 'Loading from' + load_from_file
#     print 'starting saving from checkpoint ' + str(1 + last_model_number)

# logs_path = os.path.join('..', 'tensorboard', strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

model_path = os.path.join('..', 'tensorboard', '2017_06_09_22_50_20-00000040')

if __name__ == '__main__':

    # print 'logging to', logs_path
    # train()
    
    # evaluate(model_path)

    predict(model_path)


















        