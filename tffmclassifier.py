#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: tffmclassifier.py
Author: leowan(leowan)
Date: 2018/11/16 16:14:09
"""

import math
import os
import shutil

import numpy as np
import tensorflow as tf
from scipy import sparse

class TFFMClassifier(object):
    """
        Factorization Machine Classifier
    """
    def __init__(self, feature_num=None, factor_num=10, l2_weight=0.01, batch_size=-1, input_type='dense',
        learning_rate=1e-2, epoch_num=10, random_seed=None, session_config=None,
        log_dir='/tmp/tflog/', chkpt_dir='/tmp/tfchpt/', print_step=-1):
        """
            Initializer
            Params:
                feature_num: feature number
                input_type: dense / sparse
                learning_rate: learning rate
                random_seed: random seed
        """
        self.feature_num = feature_num
        self.factor_num = factor_num
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.input_type = input_type
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.random_seed = random_seed
        self.log_dir = log_dir
        self.chkpt_dir = chkpt_dir
        self.session_config = session_config
        self.print_step = print_step

    def export_model(self, export_dir, input_type='dense'):
        """
            Export pb model
            Params:
                input_type: dense / sparse
        """
        if input_type == 'dense':
            with self.session as sess:
                builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
                builder.add_meta_graph_and_variables(sess,
                    [tf.saved_model.tag_constants.SERVING],
                    signature_def_map= {
                        "model": tf.saved_model.signature_def_utils.build_signature_def(
                            inputs= {"X": tf.saved_model.utils.build_tensor_info(self.X)},
                            outputs= {"logits": tf.saved_model.utils.build_tensor_info(self.logits)})
                        })
                builder.save(as_text=False)
        elif input_type == 'sparse':
            with self.session as sess:
                builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
                builder.add_meta_graph_and_variables(sess,
                    [tf.saved_model.tag_constants.SERVING],
                    signature_def_map= {
                        "model": tf.saved_model.signature_def_utils.build_signature_def(
                            inputs= {
                                "X_indices": tf.saved_model.utils.build_tensor_info(self.X_indices),
                                "X_values": tf.saved_model.utils.build_tensor_info(self.X_values),
                                "X_shape": tf.saved_model.utils.build_tensor_info(self.X_shape)
                            },
                            outputs= {"logits": tf.saved_model.utils.build_tensor_info(self.logits)})
                        })
                builder.save(as_text=False)

    def decision_function_imported(self, X, import_dir, input_type='dense'):
        """
            Decision function by imported Pb model
        """
        if input_type == 'dense':
            with tf.Session(graph=tf.Graph()) as sess:
                meta_graph_def = tf.saved_model.loader.load(
                    sess, [tf.saved_model.tag_constants.SERVING], import_dir)
                signature = meta_graph_def.signature_def
                X_tensor_name = signature["model"].inputs["X"].name
                logits_tensor_name = signature["model"].outputs["logits"].name
                X_tensor = sess.graph.get_tensor_by_name(X_tensor_name)
                logits_tensor = sess.graph.get_tensor_by_name(logits_tensor_name)
                return sess.run(logits_tensor, feed_dict={X_tensor: X})
        elif input_type == 'sparse':
            X_sparse = X.tocoo()
            with tf.Session(graph=tf.Graph()) as sess:
                meta_graph_def = tf.saved_model.loader.load(
                    sess, [tf.saved_model.tag_constants.SERVING], import_dir)
                signature = meta_graph_def.signature_def
                X_indices_tensor_name = signature["model"].inputs["X_indices"].name
                X_values_tensor_name = signature["model"].inputs["X_values"].name
                X_shape_tensor_name = signature["model"].inputs["X_shape"].name
                logits_tensor_name = signature["model"].outputs["logits"].name
                X_indices_tensor = sess.graph.get_tensor_by_name(X_indices_tensor_name)
                X_values_tensor = sess.graph.get_tensor_by_name(X_values_tensor_name)
                X_shape_tensor = sess.graph.get_tensor_by_name(X_shape_tensor_name)
                logits_tensor = sess.graph.get_tensor_by_name(logits_tensor_name)
                fd = {}
                fd[X_indices_tensor] = np.hstack(
                    [X_sparse.row[:, np.newaxis], X_sparse.col[:, np.newaxis]]
                ).astype(np.int64)
                fd[X_values_tensor] = X_sparse.data.astype(np.float32)
                fd[X_shape_tensor] = np.array(X_sparse.shape).astype(np.int64)
                return sess.run(logits_tensor, feed_dict=fd)

    def load_checkpoint(self, feature_num, chkpt_dir):
        """
            Load model from checkpoint
        """
        self.feature_num = feature_num
        self._build_graph()
        self._init_session()
        self.saver.restore(self.session, '{}/model'.format(chkpt_dir))

    def decision_function(self, X):
        """
            Decision function
        """
        return self.session.run(self.logits, feed_dict=self._make_feed_dict(X, None))

    def predict_proba(self, X):
        """
            Predict probabilities
        """
        logits = self.decision_function(X)
        probs_positive = self._sigmoid(logits)
        probs_negative = 1 - probs_positive
        probs = np.vstack((probs_negative.T, probs_positive.T))
        return probs.T

    def predict(self, X):
        """
            Predict
        """
        logits = self.decision_function(X)
        predictions = (logits > 0).astype(int)
        return predictions

    def fit(self, X_train, y_train):
        """
            Fit model
        """
        if self.feature_num == None:
            self.feature_num = X_train.shape[1]

        assert self.feature_num == X_train.shape[1], 'Mismatched: feature_num'

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
            batches_per_epoch = math.floor((X_train.shape[0] - 1) * 1.0 / batch_size) + 1
            best_loss = np.inf
            cur_loss = np.inf
            for X, y in self._batch(X_train, y_train, n=batch_size):
                y = np.expand_dims(y, 1)
                fd = self._make_feed_dict(X, y)
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

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._construct_placeholders()
            self._construct_weights()

            # network forward pass
            self.logits = self._forward_pass()

            # loss function
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.y)) \
                + self.l2_weight * tf.nn.l2_loss(self.b) \
                + self.l2_weight * tf.nn.l2_loss(self.w) \
                + self.l2_weight * tf.nn.l2_loss(self.v)

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
        if self.input_type == 'dense':
            self.X = tf.placeholder(shape=[None, self.feature_num],
                dtype=tf.float32, name='X')
        else:
            with tf.name_scope('sparse_placeholders') as scope:
                self.X_indices = tf.placeholder(shape=[None, 2], dtype=tf.int64, name='X_indices')
                self.X_values = tf.placeholder(shape=[None], dtype=tf.float32, name='X_values')
                self.X_shape = tf.placeholder(shape=[2], dtype=tf.int64, name='X_shape')
            self.X = tf.SparseTensor(self.X_indices, self.X_values, self.X_shape)

        self.y = tf.placeholder(shape=[None, 1],
            dtype=tf.float32, name='y')

    def _construct_weights(self):
        """
            Construct weights
        """

        self.b = tf.get_variable(name='bias', shape=[1],
            initializer=tf.truncated_normal_initializer(
            stddev=0.001, seed=self.random_seed))
        self.w = tf.get_variable(name='linear_weights',
            shape=[self.feature_num, 1],
            initializer=tf.truncated_normal_initializer(
            stddev=0.001, seed=self.random_seed))
        self.v = tf.get_variable(name='interaction_factors',
            shape=[self.feature_num, self.factor_num],
            initializer=tf.truncated_normal_initializer(
            stddev=0.001, seed=self.random_seed))

        # statistics
        tf.summary.histogram('bias', self.b)
        tf.summary.histogram('linear_weights', self.w)
        tf.summary.histogram('interaction_factors', self.v)

    def _forward_pass(self):
        """
            Forward pass
        """
        h = tf.reduce_sum(self._matmul_op(self.X, self.w, self.input_type), 1, keepdims=True) + self.b

        h = 0.5 * tf.reduce_sum(
            tf.pow(self._matmul_op(self.X, self.v, self.input_type), 2) \
            - self._matmul_op(self._pow_op(self.X, 2, self.input_type),
                tf.pow(self.v, 2), self.input_type),
            1, keepdims=True) + h

        return h

    def _matmul_op(self, a, b, input_type='dense'):
        """
            Wrapper of matmul
        """
        with tf.name_scope('matmul_op') as scope:
            if input_type == 'dense':
                return tf.matmul(a, b)
            elif input_type == 'sparse':
                return tf.sparse_tensor_dense_matmul(a, b)

    def _pow_op(self, a, p, input_type='dense'):
        """
            Wrapper of matmul
        """
        with tf.name_scope('pow_op') as scope:
            if input_type == 'dense':
                return tf.pow(a, p)
            elif input_type == 'sparse':
                return tf.SparseTensor(a.indices, tf.pow(a.values, p), a.dense_shape)

    def _batch(self, X, y, n=1):
        """
            Make batch
        """
        assert X.shape[0] == y.shape[0], 'Mismatched: dim'

        l = X.shape[0]
        for ndx in range(0, l, n):
            upper_ndx = min(ndx + n, l)
            yield (X[ndx:upper_ndx], y[ndx:upper_ndx])

    def _make_feed_dict(self, X, y):
        """
            Make feed dict
        """
        fd = {}
        if self.input_type == "dense":
            fd[self.X] = X
        else:
            X_sparse = X.tocoo()
            fd[self.X_indices] = np.hstack(
                [X_sparse.row[:, np.newaxis], X_sparse.col[:, np.newaxis]]
            ).astype(np.int64)
            fd[self.X_values] = X_sparse.data.astype(np.float32)
            fd[self.X_shape] = np.array(X_sparse.shape).astype(np.int64)
        if y is not None:
            fd[self.y] = y
        return fd

    def _sigmoid(self, x):
        """
            Sigmoid
        """
        return 1 / (1 + np.exp(-x))
