#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: tfdcnclassifier_column_indexed.py
Author: leowan(leowan)
Date: 2018/11/16 16:14:09
"""

import math
import os
import shutil

import numpy as np
import tensorflow as tf
from scipy import sparse

class TFDCNClassifier(object):
    """
        Deep Cross Network
    """
    def __init__(self,
        feature_num,
        field_num,
        factor_num=10,
        deep_layer_nodes=[32, 32],
        cross_layer_num=2,
        l2_weight=0.01, batch_size=-1,
        learning_rate=1e-2, epoch_num=10, random_seed=None, session_config=None,
        log_dir='/tmp/tflog/', chkpt_dir='/tmp/tfchpt/', print_step=-1):
        """
            Initializer
            Params:
                feature_num: feature number
                learning_rate: learning rate
                random_seed: random seed
        """
        self.feature_num = feature_num
        self.field_num = field_num
        self.factor_num = factor_num
        self.deep_layer_nodes = deep_layer_nodes
        self.cross_layer_num = cross_layer_num
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.random_seed = random_seed
        self.log_dir = log_dir
        self.chkpt_dir = chkpt_dir
        self.session_config = session_config
        self.print_step = print_step

    def export_model(self, export_dir):
        """
            Export pb model
            Params:
        """
        with self.session as sess:
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map= {
                    "model": tf.saved_model.signature_def_utils.build_signature_def(
                        inputs= {
                            "X_colind": tf.saved_model.utils.build_tensor_info(self.X_colind),
                            "X_colval": tf.saved_model.utils.build_tensor_info(self.X_colval),
                        },
                        outputs= {"logits": tf.saved_model.utils.build_tensor_info(self.logits)})
                    })
            builder.save(as_text=False)

    def decision_function_imported(self, X_colind, X_colval, import_dir):
        """
            Decision function by imported Pb model
        """
        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], import_dir)
            signature = meta_graph_def.signature_def
            X_colind_tensor_name = signature["model"].inputs["X_colind"].name
            X_colval_tensor_name = signature["model"].inputs["X_colval"].name
            logits_tensor_name = signature["model"].outputs["logits"].name
            X_colind_tensor = sess.graph.get_tensor_by_name(X_colind_tensor_name)
            X_colval_tensor = sess.graph.get_tensor_by_name(X_colval_tensor_name)
            logits_tensor = sess.graph.get_tensor_by_name(logits_tensor_name)
            return sess.run(logits_tensor, feed_dict={
                X_colind_tensor: X_colind,
                X_colval_tensor: X_colval
            })

    def load_checkpoint(self, feature_num, field_num, chkpt_dir):
        """
            Load model from checkpoint
        """
        self.feature_num = feature_num
        self.field_num = field_num
        self._build_graph()
        self._init_session()
        self.saver.restore(self.session, '{}/model'.format(chkpt_dir))

    def decision_function(self, X_colind, X_colval):
        """
            Decision function
        """
        return self.session.run(self.logits, feed_dict=self._make_feed_dict(X_colind, X_colval, None))

    def predict_proba(self, X_colind, X_colval):
        """
            Predict probabilities
        """
        logits = self.decision_function(X_colind, X_colval)
        probs_positive = self._sigmoid(logits)
        probs_negative = 1 - probs_positive
        probs = np.vstack((probs_negative.T, probs_positive.T))
        return probs.T

    def predict(self, X_colind, X_colval):
        """
            Predict
        """
        logits = self.decision_function(X_colind, X_colval)
        predictions = (logits > 0).astype(int)
        return predictions

    def fit(self, X_colind_train, X_colval_train, y_train):
        """
            Fit model
        """
        assert self.feature_num is not None, 'Null: feature_num'
        assert self.field_num is not None, 'Null: field_num'

        # clear output dirs
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        if os.path.exists(self.chkpt_dir):
            shutil.rmtree(self.chkpt_dir)
        if not os.path.isdir(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self._build_graph()
        self._init_session()

        sess = self.session

        for epoch in range(self.epoch_num):
            batch_size = self.batch_size
            if batch_size == -1:
                batch_size = X_train.shape[0]
            batch_cnt = 0
            batches_per_epoch = math.floor((y_train.shape[0] - 1) * 1.0 / batch_size) + 1
            best_loss = np.inf
            cur_loss = np.inf
            for X_colind, X_colval, y in self._batch(
                X_colind_train, X_colval_train, y_train, n=batch_size):
                y = np.expand_dims(y, 1)
                fd = self._make_feed_dict(X_colind, X_colval, y)
                _, cur_loss = sess.run([self.train_op, self.loss], feed_dict=fd)
                batch_cnt += 1
                global_step = epoch * batches_per_epoch + batch_cnt
                if self.print_step > 0 and global_step % self.print_step == 0:
                    summary_train = sess.run(self.summary_op, feed_dict=fd)
                    self.summary_writer.add_summary(summary_train, global_step=global_step)
                    print("epoch: {}, global_step: {}, loss: {}".format(
                        epoch, global_step, cur_loss))
            if cur_loss < best_loss:
                best_loss = cur_loss
        self.saver.save(sess, '{}/model'.format(self.chkpt_dir))

    def _build_graph(self):
        """
            Build graph
        """
        assert self.feature_num is not None, 'Null: feature_num'
        assert self.field_num is not None, 'Null: field_num'

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._construct_placeholders()
            self._construct_weights()

            # network forward pass
            self.logits = self._forward_pass()

            # loss function
            self.loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.y)) \
                + self.l2_weight * tf.nn.l2_loss(self.proj_w) \
                + self.l2_weight * tf.nn.l2_loss(self.proj_b)
            for i in range(len(self.deep_w)):
                self.loss += self.l2_weight * tf.nn.l2_loss(self.deep_w[i])
                self.loss += self.l2_weight * tf.nn.l2_loss(self.deep_b[i])
            for i in range(len(self.cross_w)):
                self.loss += self.l2_weight * tf.nn.l2_loss(self.cross_w[i])
                self.loss += self.l2_weight * tf.nn.l2_loss(self.cross_b[i])

            # training optimizer
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            # saver
            self.saver = tf.train.Saver()

            # initializer
            self.init_var_op = tf.global_variables_initializer()

            # statistics
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

        return self.saver, self.logits, self.loss, self.train_op, self.summary_op

    def _init_session(self):
        """
            Init session
        """
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.graph)
        self.session = tf.Session(config=self.session_config, graph=self.graph)
        self.session.run(self.init_var_op)
        print('tensorboard --logdir={} --port 8080'.format(os.path.abspath(self.log_dir)))
        #print('checkpoint dir: {}'.format(self.chkpt_dir))

    def _construct_placeholders(self):
        """
            Construct inpute placeholders
        """
        self.X_colind = tf.placeholder(shape=[None, self.feature_num],
            dtype=tf.int32, name='X_colind')
        self.X_colval = tf.placeholder(shape=[None, self.feature_num],
            dtype=tf.float32, name='X_colval')

        self.y = tf.placeholder(shape=[None, 1],
            dtype=tf.float32, name='y')

    def _construct_weights(self):
        """
            Construct weights
        """
        # embedding part
        self.embeddings = tf.get_variable(name='embeddings',
            shape=[self.feature_num, self.factor_num],
            initializer=tf.truncated_normal_initializer(
            stddev=0.001, seed=self.random_seed))
        # statistics
        tf.summary.histogram('embeddings', self.embeddings)

        # wide part
        self.cross_w = []
        self.cross_b = []
        for i in range(self.cross_layer_num):
            cross_weight_key = "cross_w_{}".format(i)
            cross_bias_key = "cross_b_{}".format(i)
            self.cross_w.append(tf.get_variable(name=cross_weight_key,
                shape=[self.field_num * self.factor_num, 1],
                initializer=tf.truncated_normal_initializer(
                stddev=0.001, seed=self.random_seed)))
            self.cross_b.append(tf.get_variable(name=cross_bias_key,
                shape=[self.field_num * self.factor_num],
                initializer=tf.truncated_normal_initializer(
                stddev=0.001, seed=self.random_seed)))

            # statistics
            tf.summary.histogram(cross_weight_key, self.cross_w[-1])
            tf.summary.histogram(cross_bias_key, self.cross_b[-1])

        # deep part
        assert len(self.deep_layer_nodes) > 0, 'Void: deep_layer_nodes'
        self.deep_w = []
        self.deep_b = []
        input_node_num = self.field_num * self.factor_num

        self.deep_w.append(tf.get_variable(
            name='deep_w_0_1', shape=[input_node_num, self.deep_layer_nodes[0]],
            initializer=tf.contrib.layers.xavier_initializer(
                seed=self.random_seed)))
        self.deep_b.append(tf.get_variable(
            name='deep_b_1', shape=[self.deep_layer_nodes[0]],
            initializer=tf.truncated_normal_initializer(
                stddev=0.001, seed=self.random_seed)))
        # statistics
        tf.summary.histogram('deep_w_0_1', self.deep_w[-1])
        tf.summary.histogram('deep_b_1', self.deep_b[-1])

        for i in range(1, len(self.deep_layer_nodes)):
            weight_key = "deep_w_{}_{}".format(i, i+1)
            bias_key = "deep_b_{}".format(i+1)
            self.deep_w.append(tf.get_variable(
                name=weight_key, shape=[self.deep_layer_nodes[i-1], self.deep_layer_nodes[i]],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            self.deep_b.append(tf.get_variable(
                name=bias_key, shape=[1],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            # statistics
            tf.summary.histogram(weight_key, self.deep_w[-1])
            tf.summary.histogram(bias_key, self.deep_b[-1])

        # final projection part
        self.proj_b = tf.get_variable(name='proj_bias', shape=[1],
            initializer=tf.truncated_normal_initializer(
            stddev=0.001, seed=self.random_seed))

        proj_node_num = self.field_num * self.factor_num + self.deep_layer_nodes[-1] # cross + deep
        self.proj_w = tf.get_variable(name='proj_w', shape=[proj_node_num, 1],
            initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))
        # statistics
        tf.summary.histogram('proj_bias', self.proj_b)
        tf.summary.histogram('proj_bias', self.proj_w)

    def _forward_pass(self):
        """
            Forward pass
        """
        # embedding layer
        embedding_t = tf.nn.embedding_lookup(self.embeddings, self.X_colind) # shaped [-1, field_num, factor_num]
        input_layer = tf.multiply(embedding_t,
            tf.reshape(self.X_colval, shape=[-1, self.field_num, 1])) # shaped [-1, field_num, factor_num]
        input_layer = tf.reshape(input_layer, shape=[-1, self.field_num * self.factor_num]) # shaped [-1, field_num * factor_num]

        # cross term
        cross_h = input_layer
        for i in range(len(self.cross_w)):
            x0_xt_w = tf.reshape(tf.matmul(cross_h, self.cross_w[i]), shape=[-1, 1]) # shaped [-1, 1]
            x0_xt_w = tf.multiply(input_layer, x0_xt_w) # shaped [-1, field_num * factor_num]
            cross_h = tf.add(cross_h, x0_xt_w)
            cross_h = tf.add(cross_h, self.cross_b[i])

        # deep term
        deep_h = input_layer
        for i in range(len(self.deep_w)):
            deep_h = tf.matmul(deep_h, self.deep_w[i]) + self.deep_b[i]
            tf.layers.batch_normalization(deep_h)
            tf.nn.relu(deep_h)

        # joint projection
        h = tf.matmul(tf.concat([cross_h, deep_h], axis=1, name='concat_wd'), self.proj_w, name='final_proj')
        h = tf.add(h, self.proj_b)

        return h

    def _batch(self, X_colind, X_colval, y, n=1):
        """
            Make batch
        """
        assert X_colind.shape[0] == y.shape[0], 'Mismatched: dim'

        l = y.shape[0]
        for ndx in range(0, l, n):
            upper_ndx = min(ndx + n, l)
            yield (X_colind[ndx:upper_ndx], X_colval[ndx:upper_ndx], y[ndx:upper_ndx])

    def _make_feed_dict(self, X_colind, X_colval, y):
        """
            Make feed dict
        """
        fd = {}
        fd[self.X_colind] = X_colind
        fd[self.X_colval] = X_colval
        if y is not None:
            fd[self.y] = y
        return fd

    def _sigmoid(self, x):
        """
            Sigmoid
        """
        return 1 / (1 + np.exp(-x))
