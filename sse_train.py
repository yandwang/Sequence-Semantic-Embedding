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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from builtins import str
from builtins import str
from builtins import range
import math
import os, logging
import random
import sys
import time
import codecs

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import sse_model
import sse_evaluator
import sse_index
import text_encoder
from data import *
import argparse
from pprint import pprint

FLAGS = None

parser = argparse.ArgumentParser(description='Sequence semantic embedding')
parser.add_argument('--loss_type', default='pruned_cross_entropy', type=str)
parser.add_argument('--q_lambda', default=0, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--bias', default=0.0, type=float)
parser.add_argument('--gamma', default=20, type=float)
parser.add_argument('--label_dims', default=571, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--learning_rate_decay_factor', default=0.99, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--embedding_size', default=50, type=int)
parser.add_argument('--encoding_size', default=80, type=int)
parser.add_argument('--src_cell_size', default=96, type=int)
parser.add_argument('--tgt_cell_size', default=96, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--vocab_size', default=32000, type=int)
parser.add_argument('--max_seq_length', default=50, type=int)
parser.add_argument('--max_epoc', default=10, type=int)
parser.add_argument('--predict_nbest', default=10, type=int)
parser.add_argument('--task_type', default='classification', type=str)
parser.add_argument('--data_dir', default='rawdata-classification', type=str)
parser.add_argument('--model_dir', default='models-classification/top10_epoch10', type=str)
parser.add_argument('--rawfilename', default='targetIDs', type=str)
parser.add_argument('--encodedIndexFile', default='targetEncodingIndex.tsv', type=str)
parser.add_argument('--device', default='1', type=str)
parser.add_argument('--network_mode', default='dual-encoder', type=str)
parser.add_argument('--steps_per_checkpoint', default=200, type=int)

args = parser.parse_args()
pprint(vars(args))


# tf.app.flags.DEFINE_string("loss_type", 'pruned_cross_entropy', "Loss type.")
# tf.app.flags.DEFINE_float("q_lambda", 0, "q_lambda")
# tf.app.flags.DEFINE_float("alpha", .5, "alpha")
# tf.app.flags.DEFINE_float("bias", 0.0, "bias")
# tf.app.flags.DEFINE_float("gamma", 20, "gamma")
# tf.app.flags.DEFINE_integer("label_dims", 571, "label_dims of eBay CSA dataset")

# tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
#                           "Learning rate decays by this much.")

# tf.app.flags.DEFINE_integer("batch_size", 64,
#                             "Batch size to use during training(positive pair count based).")
# tf.app.flags.DEFINE_integer("embedding_size", 50, "Size of word embedding vector.")
# tf.app.flags.DEFINE_integer("encoding_size", 80,
#                             "Size of sequence encoding vector. Same number of nodes for each model layer.")
# tf.app.flags.DEFINE_integer("src_cell_size", 96, "LSTM cell size in source RNN model.")
# tf.app.flags.DEFINE_integer("tgt_cell_size", 96,
#                             "LSTM cell size in target RNN model. Same number of nodes for each model layer.")
# tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
# tf.app.flags.DEFINE_integer("vocab_size", 32000,
#                             "If no vocabulary file provided, will use this size to build vocabulary file from training data.")
# tf.app.flags.DEFINE_integer("max_seq_length", 50, "max number of words in each source or target sequence.")
# tf.app.flags.DEFINE_integer("max_epoc", 10, "max epoc number for training procedure.")
# tf.app.flags.DEFINE_integer("predict_nbest", 10, "max top N for evaluation prediction.")
# tf.app.flags.DEFINE_string("task_type", 'classification',
#                            "Type of tasks. We provide data, training receipe and service demos for four different type tasks:  classification, ranking, qna, crosslingual")
#
# tf.app.flags.DEFINE_string("data_dir", 'rawdata-classification', "Data directory")
# tf.app.flags.DEFINE_string("model_dir", 'models-classification/top10_epoch10', "Trained model directory.")
# tf.app.flags.DEFINE_string("rawfilename", 'targetIDs', "raw target sequence file to be indexed")
# tf.app.flags.DEFINE_string("encodedIndexFile", 'targetEncodingIndex.tsv', "target sequece encoding index file.")
#
# tf.app.flags.DEFINE_string("device", "0",
#                            "Default to use GPU:0. Softplacement used, if no GPU found, further default to cpu:0.")
# tf.app.flags.DEFINE_string("network_mode", 'dual-encoder',
#                            "Setup SSE network configration mode. SSE support three types of modes: source-encoder-only, dual-encoder, shared-encoder.")
# tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
#                             "How many training steps to do per checkpoint.")

# FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # value can be 0,1,2, 3
# os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device  # value can be 0,1,2, 3


def create_model(session, targetSpaceSize, vocabsize, forward_only):
    """Create SSE model and initialize or load parameters in session."""

    # modelParams = {'max_seq_length': args.max_seq_length, 'vocab_size': vocabsize,
    #                'embedding_size': args.embedding_size, 'encoding_size': args.encoding_size,
    #                'learning_rate': args.learning_rate, 'learning_rate_decay_factor': args.learning_rate_decay_factor,
    #                'src_cell_size': args.src_cell_size, 'tgt_cell_size': args.tgt_cell_size,
    #                'network_mode': args.network_mode, 'predict_nbest': args.predict_nbest,
    #                'targetSpaceSize': targetSpaceSize, 'forward_only': forward_only,
    #                'loss_type': args.loss_type, 'q_lambda': args.q_lambda,
    #                'alpha': args.alpha, 'bias': args.bias, 'gamma': args.gamma,
    #                'label_dims': args.label_dims}

    data_utils.save_model_configs(args.model_dir, args)

    model = sse_model.SSEModel(args)

    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if forward_only:
            print('Error!!!Could not load any model from specified folder: %s' % args.model_dir)
            exit(-1)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
    return model


def set_up_logging(run_id):
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("%s.log" % run_id)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def train():
    # Prepare data.
    print("Preparing Train & Eval data in %s" % args.data_dir)

    for d in args.data_dir, args.model_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    data = Data(args.model_dir, args.data_dir, args.vocab_size, args.max_seq_length)
    epoc_steps = len(data.rawTrainPosCorpus) // args.batch_size

    print("Training Data: %d total positive samples, each epoch need %d steps" % (
    len(data.rawTrainPosCorpus), epoc_steps))

    cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=cfg) as sess:
        model = create_model(sess, data.rawnegSetLen, data.vocab_size, False)

        # setup tensorboard logging
        sw = tf.summary.FileWriter(logdir=args.model_dir, graph=sess.graph, flush_secs=120)
        # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # summary_op_te = tf.summary.merge(summaries)
        summary_op = model.add_summaries()

        # This is the training loop.
        step_time, loss, train_acc = 0.0, 0.0, 0.0
        current_step = 0
        previous_accuracies = []
        for epoch in range(args.max_epoc):
            epoc_start_Time = time.time()

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            tf.summary.scalar("lr", model.learning_rate.eval())
            tf.get_collection(tf.GraphKeys.SUMMARIES)
            # Build the summary operation from the last tower summaries.
            summary_op_te = tf.summary.merge_all()

            for batchId in range(epoc_steps):
                start_time = time.time()
                source_inputs, tgt_inputs, labels = data.get_train_batch(args.batch_size)
                model.set_forward_only(False)
                d = model.get_train_feed_dict(source_inputs, tgt_inputs, labels)
                ops = [model.train, summary_op_te, model.loss, model.train_acc]
                _, summary, step_loss, step_train_acc = sess.run(ops, feed_dict=d)
                step_time += (time.time() - start_time) / args.steps_per_checkpoint
                loss += step_loss / args.steps_per_checkpoint
                train_acc += step_train_acc / args.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % args.steps_per_checkpoint == 0:
                    print(
                        "global epoc: %.3f, global step %d, learning rate %.4f step-time:%.2f loss:%.4f train_binary_acc:%.4f " %
                        (float(model.global_step.eval()) / float(epoc_steps), model.global_step.eval(),
                         model.learning_rate.eval(),
                         step_time, step_loss, train_acc))

                    sw.add_summary(summary, current_step)

                    checkpoint_path = os.path.join(args.model_dir, "SSE-LSTM.ckpt")
                    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="train_binary_acc", simple_value=train_acc)])
                    sw.add_summary(acc_sum, current_step)

                    # #########debugging##########
                    # model.set_forward_only(True)
                    # sse_index.createIndexFile(model, encoder, os.path.join(FLAGS.model_dir, FLAGS.rawfilename),
                    #                           FLAGS.max_seq_length, os.path.join(FLAGS.model_dir, FLAGS.encodedIndexFile), sess,
                    #                           batchsize=1000)
                    # evaluator = sse_evaluator.Evaluator(model, eval_corpus, os.path.join(FLAGS.model_dir, FLAGS.encodedIndexFile),
                    #                                     sess)
                    # acc1, acc3, acc10 = evaluator.eval()
                    # print("epoc# %.3f, task specific evaluation: top 1/3/10 accuracies: %f / %f / %f " % (float(model.global_step.eval())/ float(epoc_steps), acc1, acc3, acc10))
                    # ###end of debugging########

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_accuracies) > 3 and train_acc < min(previous_accuracies[-2:]):
                        sess.run(model.learning_rate_decay_op)

                    previous_accuracies.append(train_acc)
                    tf.summary.scalar('training_accuracy', train_acc)
                    # save currently best-ever model
                    if train_acc == max(previous_accuracies):
                        print("Better Accuracy %.4f found. Saving current best model ..." % train_acc)
                        model.save(sess, checkpoint_path + "-BestEver")
                    else:
                        print("Best Accuracy is: %.4f, while current round is: %.4f" % (
                        max(previous_accuracies), train_acc))
                        print("skip saving model ...")
                    # if finished at least 2 Epocs and still no further accuracy improvement, stop training
                    # report the best accuracy number and final model's number and save it.
                    if epoch > 10 and train_acc < min(previous_accuracies[-5:]):
                        p = model.save(sess, checkpoint_path + "-final")
                        print(
                            "After around %d Epocs no further improvement, Training finished, wrote checkpoint to %s." % (
                            epoch, p))
                        break

                    # reset current checkpoint step statistics
                    step_time, loss, train_acc = 0.0, 0.0, 0.0

            epoc_train_time = time.time() - epoc_start_Time
            print('\n\n\nepoch# %d  took %f hours' % (epoch, epoc_train_time / (60.0 * 60)))

            # run task specific evaluation afer each epoch
            if (args.task_type not in ['ranking', 'crosslingual']) or ((epoch + 1) % 20 == 0):
                model.set_forward_only(True)
                sse_index.createIndexFile(model, data.encoder, os.path.join(args.model_dir, args.rawfilename),
                                          args.max_seq_length, os.path.join(args.model_dir, args.encodedIndexFile),
                                          sess, batchsize=1000)
                evaluator = sse_evaluator.Evaluator(model, data.rawEvalCorpus,
                                                    os.path.join(args.model_dir, args.encodedIndexFile), sess)
                acc1, acc3, acc10 = evaluator.eval()
                print("epoc#%d, task specific evaluation: top 1/3/10 accuracies: %f / %f / %f \n\n\n" % (
                epoch, acc1, acc3, acc10))
            # Save checkpoint at end of each epoch
            checkpoint_path = os.path.join(args.model_dir, "SSE-LSTM.ckpt")
            model.save(sess, checkpoint_path + '-epoch-%d' % epoch)
            if len(previous_accuracies) > 0:
                print('So far best ever model training binary accuracy is: %.4f ' % max(previous_accuracies))


def main():
    global FLAGS
    parser = argparse.ArgumentParser()

    if 'KRYLOV_DATA_DIR' not in os.environ:
        os.environ['KRYLOV_DATA_DIR'] = ''
    if 'KRYLOV_WF_PRINCIPAL' not in os.environ:
        os.environ['KRYLOV_WF_PRINCIPAL'] = ''

    data_dir = os.path.join(os.environ['KRYLOV_DATA_DIR'], os.environ['KRYLOV_WF_PRINCIPAL'], 'rawdata-classification')
    model_dir = os.path.join(os.environ['KRYLOV_DATA_DIR'], os.environ['KRYLOV_WF_PRINCIPAL'], 'top10')

    print(data_dir)
    print(model_dir)

    parser.add_argument('--data_dir', type=str,
                        default=data_dir,
                        help='Directory for storing input data')
    parser.add_argument('--model_dir', type=str,
                        default=model_dir,
                        help='Directory for storing output data')

    FLAGS, unparsed = parser.parse_known_args()
    run_id = 'BatchSize' + str(args.batch_size) + '.EmbedSize' + str(args.embedding_size) + \
             '.EncodeSize' + str(args.encoding_size) + '.SrcCell' + str(args.src_cell_size) + \
             '.TgtCell' + str(args.tgt_cell_size) + '.SrcCell' + str(args.src_cell_size) + \
             '.' + str(args.network_mode) + \
             '.' + str(time.time())[-5:]
    set_up_logging(run_id)
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)

if __name__ == "__main__":
    FLAGS = None
    main()

