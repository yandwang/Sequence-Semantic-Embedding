# coding=utf-8
################################################################################
#
# Copyright (c) 2016 eBay Software Foundation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#################################################################################
#
# @Author: Mingkuan Liu
# @Email:  mingkliu@ebay.com
# @Date:   2016-07-24
#
##################################################################################

"""

SSE (Sequence Semantic Embedding) is an encoder framework toolkit for NLP related tasks and it's implemented in
TensorFlow by leveraging TF's convenient DNN/CNN/RNN/LSTM building blocks.

SSE model translates a sequence of symbols into a vector of numeric numbers, so that different sequences
with similar semantic meanings will have closer numeric vector distances. This numeric number vector is
called the SSE for the original sequence of symbols.

SSE can be applied to some large scale NLP related machine learning tasks. For example, it can be applied to
large scale classification task: mapping a listing title to one of the 20,000+ leaf categories in eBay website.
Or it can be applied to information retrieval task: mapping a search query to some most relevant documents in the inventory.

Depending on each specific task, similar semantic meanings can have different definitions. For example,
in a classification task, similar semantic meanings means that for each correct pair of (listing-title, category),
the SSE of title is close to the SSE of corresponding category.  While in an information retrieval task,
similar semantic meaning means for each relevant pair of (query, document), the SSE of query is close to the SSE of
relevant document.


SSE encoder framework supports three different types of network configuration modes: source-encoder-only, dual-encoder
and shared-encoder.

* In source-encoder-only mode, SSE will only train a single encoder model(RNN/LSTM/CNN) for source sequence.
For target sequence, SSE will just learn its sequence embedding directly without applying any encoder models.
This mode is suitable for closed target space tasks such as classification task, since in such tasks the target
sequence space is limited and closed thus it does not require to generate new embeddings for any future unknown
target sequences outside of training stage.

* In dual-encoder mode, SSE will train two different encoder models(RNN/LSTM/CNN) for both source sequence and
target sequence. This mode is suitable for open target space tasks such as information retrieval, since the target
sequence space in those tasks is open and dynamically changing, a specific target sequence encoder model is needed to
generate embeddings for new unobserved target sequence outside of training stage.

* In shared-encoder mode, SSE will train one single encoder model(RNN/LSTM/CNN) shared for both source sequence
and target sequence. This mode is suitable for open target space tasks such as question answering system or
information retrieval system, since the target sequence space in those tasks is open and dynamically changing,
a specific target sequence encoder model is needed to generate embeddings for new unobserved target sequence
outside of training stage. In shared-encoder mode, the source sequence encoder model is the same as target
sequence encoder mode. Thus this mode is better for tasks where the vocabulary between source sequence and
target sequence are similar and can be shared.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import zip
from builtins import range
from builtins import object
import random
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
from tensorflow.contrib.rnn import stack_bidirectional_rnn


class SSEModel(object):

    # Number of target sequences returned by prediction operation

    def __init__(self, modelParams):

        """ Create the Sequence Semantic Embedding Model.

    :param vocab_size:  data vocab size. Note both source and target sequence share same vocabulary
    :param word_embed_size: dim for token level word embedding
    :param seq_embed_size:  dim for sequence level embedding
    :param src_cell_size:  source model RNN cell size
    :param tgt_cell_size:  target model RNN cell size
    :param num_layers:  number of layers for src/target model.
    :param learning_rate: learning rate to start with.
    :param learning_rate_decay_factor: decay learning rate by this much when needed
    :param network_mode: Config SSE network mode to be one of three modes: source-encoder-only, dual-encoder or shared-encoder.
    """

        # List of operations to be called after each training step, see
        # _add_post_train_ops
        self._post_train_ops = []
        self.name = 'SSEmodel'
        self.forward_only = bool(modelParams['forward_only'])
        self.network_mode = modelParams['network_mode']
        self.TOP_N = int(modelParams['predict_nbest'])
        self.MAX_SEQ_LENGTH = int(modelParams['max_seq_length'])
        self.max_gradient_norm = 5.0
        self.vocab_size = int(modelParams['vocab_size'])
        self.word_embed_size = int(modelParams['embedding_size'])
        self.seq_embed_size = int(modelParams['encoding_size'])
        self.src_cell_size = int(modelParams['src_cell_size'])
        self.tgt_cell_size = int(modelParams['tgt_cell_size'])
        self.learning_rate = tf.Variable(float(modelParams['learning_rate']), name='learning_rate', trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            tf.maximum(self.learning_rate * float(modelParams['learning_rate_decay_factor']), 1e-4))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.targetSpaceSize = int(modelParams['targetSpaceSize'])
        self.loss_type = modelParams['loss_type']
        self.lr = modelParams['learning_rate']
        self.q_lambda = modelParams['q_lambda']
        self.alpha = modelParams['alpha']
        self.bias = modelParams['bias']
        self.gamma = modelParams['gamma']

        # setup basic model cell type to be LSTM or GRU or CNN
        # TODO: enhence for CNN basic unit later

        # Setup Source internal RNN Cell in tensoflow graph
        self._create_embedders()

        with tf.name_scope('stage'):
            # 0 for training, 1 for validation
            self.stage = tf.placeholder_with_default(tf.constant(0), [])

        self.train_op = self._def_loss()

        ### Setup session
        print ("launching session")
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        self.global_step = tf.Variable(0, trainable=False)
        # self.train_op = self.apply_loss_function(self.global_step)
        self.sess.run(tf.global_variables_initializer())

        self._def_optimize()
        self._def_predict()

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

        ### Initialize setting
        print ("initializing")
        np.set_printoptions(precision=4)

    @staticmethod
    def _last_relevant(output, length):
        b_size = tf.shape(output)[0]
        max_len = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        idx = tf.range(0, b_size) * max_len + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, idx)
        return relevant

    def _create_embedders(self):

        # placeholder for input data
        self._src_input_data = tf.placeholder(tf.int32, [None, self.MAX_SEQ_LENGTH], name='source_sequence')
        self._tgt_input_data = tf.placeholder(tf.int32, [None, self.MAX_SEQ_LENGTH], name='target_sequence')
        self._labels = tf.placeholder(tf.float32, [None], name='targetSpace_labels')


        # create word embedding vectors
        # note: both source and target sequence share same vocabulary and word embeddings
        self.word_embedding = tf.get_variable('word_embedding', [self.vocab_size, self.word_embed_size],
                                              initializer=tf.random_uniform_initializer(-0.25, 0.25))

        # transform input tensors from tokenID to word embedding
        self.src_input_distributed = tf.nn.embedding_lookup(self.word_embedding, self._src_input_data,
                                                            name='dist_source')
        self.tgt_input_distributed = tf.nn.embedding_lookup(self.word_embedding, self._tgt_input_data,
                                                            name='dist_target')

        if self.network_mode == 'source-encoder-only':
            self._source_encoder_only_network()
        elif self.network_mode == 'dual-encoder':
            self._dual_encoder_network()
        elif self.network_mode == 'shared-encoder':
            self._shared_encoder_network()
        elif self.network_mode == 'source_only_cnn':
            self._cnn_encoder_network()
        else:
            print(
                'Error!! Unsupported network mode: %s. Please specify on: source-encoder-only, dual-encoder or shared-encoder.' % self.network_mode)
            exit(-1)

    def _cnn_encoder_network(self):
        with tf.variable_scope('source_only_cnn'):
            self.src_input_expanded = tf.expand_dims(self.src_input_distributed, -1)
            pooled_outputs = []
            filter_sizes = [2, 3, 4, 5]
            num_filters = [256, 128, 128, 64]
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-{0}'.format(filter_size)):
                    filter_shape = [filter_size, self.word_embed_size, 1, num_filters[i]]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters[i]]), name='b')
                    conv = tf.nn.conv2d(
                        self.src_input_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pool = tf.nn.max_pool(
                        h,
                        ksize=[1, self.MAX_SEQ_LENGTH - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                    pooled_outputs.append(pool)

            feature_size = sum(num_filters)
            self.pool = tf.concat(3, pooled_outputs)
            self.pool_flat = tf.reshape(self.pool, [-1, feature_size])

            self.src_M = tf.get_variable('src_M', shape=[feature_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())
            self.src_seq_embedding = tf.matmul(self.pool_flat, self.src_M)

        with tf.variable_scope('target_embedding'):
            self.tgt_seq_embedding = tf.get_variable('tgt_seq_embedding',
                                                     shape=[self.targetSpaceSize, self.seq_embed_size],
                                                     initializer=tf.random_uniform_initializer(-0.25, 0.25))

    def _source_encoder_only_network(self):
        # config SSE network to be source encoder only mode
        # Build source encoder
        with tf.variable_scope('source_only_encoder'):
            # TODO: need play with forgetGate and peeholes here
            src_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.src_cell_size)
            src_output, _ = tf.contrib.rnn.static_rnn(src_cell, tf.unstack(self.src_input_distributed, axis=1),
                                                      dtype=tf.float32)
            src_last_output = src_output[-1]

            self.src_M = tf.get_variable('src_M', shape=[self.src_cell_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())
            self.src_seq_embedding = tf.matmul(src_last_output, self.src_M)

        # Build target encoder
        with tf.variable_scope('target_embedding'):
            # no need train target model, just train embedding directly
            self.tgt_seq_embedding = tf.get_variable('tgt_seq_embedding',
                                                     shape=[self.targetSpaceSize, self.seq_embed_size],
                                                     initializer=tf.random_uniform_initializer(-0.25, 0.25))

    def _dual_encoder_network(self):
        # config SSE network to be dual encoder mode
        # Build source encoder
        with tf.variable_scope('source_encoder'):
            src_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.src_cell_size)
            src_output, _ = tf.contrib.rnn.static_rnn(src_cell, tf.unstack(self.src_input_distributed, axis=1),
                                                      dtype=tf.float32)
            src_last_output = src_output[-1]

            self.src_M = tf.get_variable('src_M', shape=[self.src_cell_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())
            self.src_seq_embedding = tf.matmul(src_last_output, self.src_M)

            # squash the continuous code to be within [−1, 1]
            src_emd_before = tf.summary.histogram("src_embedding_before", self.src_seq_embedding)
            self.src_seq_embedding = tf.tanh(self.src_seq_embedding)
            src_emd_after = tf.summary.histogram("src_embedding_after", self.src_seq_embedding)
            
            ## Added by Yandan
            self.img_last_layer_src = self.src_seq_embedding
            self.src_seq_embedding = tf.sign(self.src_seq_embedding)
            singed_src_emd = tf.summary.histogram("signed_src_embedding", self.src_seq_embedding)

        # Build target encoder
        with tf.variable_scope('target_encoder'):
            tgt_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.tgt_cell_size)
            tgt_output, _ = tf.contrib.rnn.static_rnn(tgt_cell, tf.unstack(self.tgt_input_distributed, axis=1),
                                                      dtype=tf.float32)
            tgt_last_output = tgt_output[-1]

            self.tgt_M = tf.get_variable('tgt_M', shape=[self.tgt_cell_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())
            self.tgt_seq_embedding = tf.matmul(tgt_last_output, self.tgt_M)

            # squash the continuous code to be within [−1, 1]
            self.tgt_seq_embedding = tf.tanh(self.tgt_seq_embedding)

            ## Added by Yandan
            self.img_last_layer_tgt = self.tgt_seq_embedding
            self.tgt_seq_embedding = tf.sign(self.tgt_seq_embedding)
            singed_tgt_emd = tf.summary.histogram("signed_tgt_embedding", self.tgt_seq_embedding)

    def _shared_encoder_network(self):
        # config SSE network to be shared encoder mode
        # Build shared encoder
        with tf.variable_scope('shared_encoder'):
            src_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.src_cell_size)
            src_output, _ = tf.contrib.rnn.static_rnn(src_cell, tf.unstack(self.src_input_distributed, axis=1),
                                                      dtype=tf.float32)
            src_last_output = src_output[-1]
            self.src_M = tf.get_variable('src_M', shape=[self.src_cell_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())
            self.src_seq_embedding = tf.matmul(src_last_output, self.src_M)
            # declare tgt_M tensor before reuse them
            self.tgt_M = tf.get_variable('tgt_M', shape=[self.src_cell_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())

        with tf.variable_scope('shared_encoder', reuse=True):
            tgt_output, _ = tf.contrib.rnn.static_rnn(src_cell, tf.unstack(self.tgt_input_distributed, axis=1),
                                                      dtype=tf.float32)
            tgt_last_output = tgt_output[-1]
            self.tgt_seq_embedding = tf.matmul(tgt_last_output, self.tgt_M)

    def _def_loss(self):
        # compute src / tgt similarity
        with tf.variable_scope('similarity'):
            self.norm_src_seq_embedding = tf.nn.l2_normalize(self.img_last_layer_src, dim=-1)
            self.norm_tgt_seq_embedding = tf.nn.l2_normalize(self.img_last_layer_tgt, dim=-1)
            
            # this similarity tensor is used for prediction, tensor shape is (src_batch_size * target_space_size )
            # self.similarity = tf.matmul( self.norm_src_seq_embedding, self.norm_tgt_seq_embedding, transpose_b=True)
            self.similarity = tf.matmul(self.src_seq_embedding, self.tgt_seq_embedding, transpose_b=True)
            # self.similarity = tf.Print(self.similarity, [self.similarity], summarize=571, message='similarity')

            # self.norm_similarity = tf.matmul( tf.nn.l2_normalize(self.src_seq_embedding, dim=-1),
            # tf.nn.l2_normalize( self.tgt_seq_embedding, dim=-1), transpose_b=True) this binary logit tensor is used
            #  for training, tensor shape is (src_batch_size * 1)

            # self.binarylogit =  tf.reduce_sum( tf.multiply(self.norm_src_seq_embedding,
            # self.norm_tgt_seq_embedding) , axis=-1 )
            self.binarylogit = tf.reduce_sum(tf.multiply(self.src_seq_embedding, self.tgt_seq_embedding), axis=-1)
            bin_logit = tf.summary.histogram("binarylogit", self.binarylogit)
            sigmoid_binary_logit = tf.sigmoid(self.binarylogit)
            sig_bin_logit = tf.summary.histogram("sigmoid_binarylogit", sigmoid_binary_logit)
            #print(self.binarylogit.eval())
            #print(self.src_seq_embedding.eval())
            #tf.Print(self.tgt_seq_embedding, [self.tgt_seq_embedding], 'self.tgt_seq_embedding')
            # self.binarylogit = tf.Print(self.binarylogit, [self.binarylogit], summarize=6, message='binarylogit')

        with tf.variable_scope('training_loss'):
            # TODO: try logistic Binary cross entropy loss function later: tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
            # basic bianry logistic loss, treat pos and neg the same weight
            # self.loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=self.binarylogit, labels= self._labels) )
            # weighted loss, to treat pos/neg loss with different weight
            # #self.loss = tf.reduce_mean(
            ##   tf.nn.weighted_cross_entropy_with_logits(logits=self.binarylogit, targets=self._labels, pos_weight=1.0))
            # self.loss = tf.Print(self.loss, [self.loss], summarize=6, message='loss')
            # self.loss = tf.reduce_mean(tf.multiply(self._labels, 1.0 - tf.sigmoid(self.binarylogit)))   +  tf.reduce_mean(tf.multiply(1 - self._labels, tf.sigmoid(self.binarylogit) ))

            # Cauchy cross-entropy loss
            if self.loss_type == 'cross_entropy':
                self.cos_loss = self.cross_entropy(self.img_last_layer_src, self.img_last_layer_tgt, self._labels, self.alpha, False, False,
                                                   self.bias)
            elif self.loss_type == 'normed_cross_entropy':
                self.cos_loss = self.cross_entropy(self.img_last_layer_src, self.img_last_layer_tgt, self._labels, self.alpha, True, False,
                                                   self.bias)
            elif self.loss_type == 'pruned_cross_entropy':
                self.cos_loss = self.cross_entropy(self.img_last_layer_src, self.img_last_layer_tgt, self._labels, self.alpha, False, True,
                                                   self.bias)
            elif self.loss_type == 'pruned_normed_cross_entropy':
                self.cos_loss = self.cross_entropy(self.img_last_layer_src, self.img_last_layer_tgt, self._labels, self.alpha, True, True,
                                                   self.bias)

            # Cauchy quantization loss
            d_src_1 = tf.square(
                tf.norm(
                    tf.subtract(tf.abs(self.norm_src_seq_embedding),
                                tf.nn.l2_normalize(
                                    tf.ones_like(self.norm_src_seq_embedding)))))
            d_tgt_1 = tf.square(
                tf.norm(
                    tf.subtract(tf.abs(self.norm_tgt_seq_embedding),
                                tf.nn.l2_normalize(
                                    tf.ones_like(self.norm_tgt_seq_embedding)))))
            self.Cauchy_quan_src = tf.reduce_mean(tf.log(tf.add(1.0, tf.div(d_src_1, self.gamma))))
            self.Cauchy_quan_tgt = tf.reduce_mean(tf.log(tf.add(1.0, tf.div(d_tgt_1, self.gamma))))
            self.Cauchy_quan_loss = self.q_lambda *(self.Cauchy_quan_src + self.Cauchy_quan_tgt)

            # Total loss
            self.loss = self.cos_loss + self.Cauchy_quan_loss

            # compute the binary training accuracy
            self.train_acc = tf.reduce_mean(
                tf.multiply(self._labels, tf.floor(tf.sigmoid(self.binarylogit) + 0.2))) + tf.reduce_mean(
                tf.multiply(1.0 - self._labels, tf.floor(1.2 - tf.sigmoid(self.binarylogit))))

        ########## Testing with Siamese loss with margin ############
        # with tf.variable_scope('training_loss'):
        #   # siamese loss with margin
        #   margin = 0.25
        #   # Calculate the positive losses
        #   pos_loss_term = 0.25 * tf.square(tf.subtract(1., self.binarylogit))
        #   pos_mult = tf.cast(self._labels, tf.float32)
        #   positive_loss = tf.multiply(pos_mult, pos_loss_term)
        #   # Calculate negative losses, then make sure on dissimilar strings
        #   neg_mult = tf.subtract(1., tf.cast(self._labels, tf.float32))
        #   negative_loss = neg_mult * tf.square(self.binarylogit)
        #   # Combine positive and negative losses
        #   loss = tf.add(positive_loss, negative_loss)
        #   # Create the margin term.  This is when the targets are 0.,
        #   #  and the scores are less than m, return 0.
        #   # Check if target is zero (dissimilar strings)
        #   target_zero = tf.equal(tf.cast(self._labels, tf.float32), 0.)
        #   # Check if cosine outputs is smaller than margin
        #   less_than_margin = tf.less(self.binarylogit, margin)
        #   # Check if both are true
        #   both_logical = tf.logical_and(target_zero, less_than_margin)
        #   both_logical = tf.cast(both_logical, tf.float32)
        #   # If both are true, then multiply by (1-1)=0.
        #   multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
        #   total_loss = tf.multiply(loss, multiplicative_factor)
        #   # Average loss over batch
        #   self.loss = tf.reduce_mean(total_loss)
        #   #compute the binary training accuracy
        #   self.train_acc = tf.reduce_mean(tf.multiply(self._labels, tf.floor(tf.sigmoid(self.binarylogit) + 0.2) ))   +  tf.reduce_mean(tf.multiply(1 - self._labels, tf.floor(1.2 - tf.sigmoid(self.binarylogit))) )

    # Cross_entropy for deep hashing
    def cross_entropy(self, img_last_layer_src, img_last_layer_tgt, label_u, alpha=0.5, normed=False, pruned=True, bias=0.0):

        # compute normalized Euclidean distance between src_seq_embedding and tgt_seq_embedding
        seq_shape = tf.shape(img_last_layer_src)
        K = seq_shape[1]
        sum_all = seq_shape[0]
        euc_norm = tf.square(tf.norm(self.norm_src_seq_embedding - self.norm_tgt_seq_embedding, ord='euclidean', axis=1))
        d_src_tgt = 0.25*tf.multiply(tf.cast(K, tf.float32), euc_norm)

        # compute parts of Cauchy cross-entropy loss
        log1 = tf.log(tf.div(d_src_tgt, self.gamma))
        log1_s = tf.multiply(label_u, log1)
        log2 = tf.log(1.0+tf.div(self.gamma, d_src_tgt+1e-8))

        # Compute the data imbalance parameters
        sum1 = tf.reduce_sum(label_u)
        balance_p = tf.div(tf.cast(sum_all, tf.float32), sum1)
        balance_n = tf.div(tf.cast(sum_all, tf.float32), tf.cast(sum_all, tf.float32)-sum1)
        balance_par = tf.multiply(balance_p, label_u) + tf.multiply(balance_n, (1.0-label_u))

        # Return the final Cauchy cross entropy
        Cauchy_entropy = tf.reduce_mean(tf.multiply(balance_par, tf.add(log1_s, log2)))
        return Cauchy_entropy

    def set_top_n(self, top_n):
        self.TOP_N = top_n

    def set_forward_only(self, forward_only=True):
        self.forward_only = forward_only

    def _def_predict(self):
        # Prediction cannot return more candidates than there are categories
        # top_n = min( tf.shape(self._tgt_input_data)[0], SSEModel.TOP_N)
        with tf.name_scope("prediction"):
            self.predicted_tgts_score, self.predicted_labels = tf.nn.top_k(self.similarity, self.TOP_N, sorted=True)
            # normalize conf score
            self.predicted_tgts_score = tf.nn.l2_normalize(self.predicted_tgts_score, 1)
            # self.predicted_tgts_score, self.predicted_labels = tf.nn.top_k(self.norm_similarity, self.TOP_N, sorted=True)
            # self.predictResults = self.similarity

    def _def_optimize(self):
        """
    Builds graph to minimize loss function.
    """
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_gradient_norm)
        self.train = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=self.global_step)
        self._add_post_train_ops()

    def _add_post_train_ops(self):
        """
    Replaces the self.train operation with an operation group, consisting of
    the training operation itself and the operations listed in
    self.post_train_ops.

    Called by _def_optimize().

    """
        with tf.control_dependencies([self.train]):
            self.train = tf.group(self.train, *self._post_train_ops)

    def save(self, session, path, global_step=None):
        """ Saves variables to given path """
        return self.saver.save(session, path, global_step)

    def load(self, session, path):
        """ Restores variables from given path """
        self.saver.restore(session, path)

    def add_summaries(self):
        """
    Adds summaries for the following variables to the graph and returns
    an operation to evaluate them.
     * loss (raw)
     * loss (moving average)

    """
        # his = tf.summary.histogram("weights", self)
        # lr = tf.summary.scalar("lr", self.learning_rate)
        Cauchy_cross_entropy =tf.summary.scalar('Cauchy_cross_entropy', self.cos_loss)
        Cauchy_quan_loss = tf.summary.scalar('Cauchy_quan_loss', self.Cauchy_quan_loss)
        loss = tf.summary.scalar('Total_loss', self.loss)
        # loss = tf.summary.scalar("loss_raw", self.loss)
        return tf.summary.merge([Cauchy_cross_entropy, Cauchy_quan_loss, loss])

    def get_predict_feed_dict(self, srcSeqs, tgtSeqs):
        """
    Returns a batch feed dict for given srcSequences passed as
    [batch_size, srcSequenceTokenIds].

    """
        d = {}
        d[self._src_input_data] = np.array(srcSeqs, dtype=np.int32)
        d[self._tgt_input_data] = np.array(tgtSeqs, dtype=np.int32)
        return d

    def get_train_feed_dict(self, srcSeqs, tgtSeqs, labels):
        """
    Returns a batch feed dict for given srcSquence and tgtSequences.

    """
        d = {}
        d[self._src_input_data] = np.array(srcSeqs, dtype=np.int32)
        d[self._labels] = np.array(labels, dtype=np.float32)
        d[self._tgt_input_data] = np.array(tgtSeqs, dtype=np.int32)
        return d

    def get_source_encoding_feed_dict(self, srcSeqs):
        """
    Returns a batch feed dict for given srcSquences.

    """
        d = {}
        d[self._src_input_data] = np.array(srcSeqs, dtype=np.int32)
        return d

    def get_target_encoding_feed_dict(self, tgtSeqs):
        """
    Returns a batch feed dict for given tgtSequences.

    """
        d = {}
        d[self._tgt_input_data] = np.array(tgtSeqs, dtype=np.int32)
        return d

    # def hamming_distance(self):
    #     """
    # Returns the hamming distance of two binary vectors
    #
    #         """
    #     binary_source_embedding = tf.sign(self.source_embedding)
    #     binary_target_embedding = tf.sign(self.target_embedding)
    #     xor_src_tag = tf.logical_xor(binary_source_embedding, binary_target_embedding)
    #     d = tf.count_nonzero(xor_src_tag)
    #     return d
