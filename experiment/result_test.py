# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from keras.preprocessing import sequence
import sys
import os
from six.moves import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.Evaluation import precision_each_class
from evaluation.Evaluation import recall_each_class
from evaluation.Evaluation import f1_each_class_precision_recall
from evaluation.Evaluation import class_label_count
from evaluation.Evaluation import print_out
from model.Classifier import Model


if __name__ == '__main__':
    print('load data........')
    with open('./../resource/generated/input.pickle', 'rb') as f:
        parameter = pickle.load(f)
        wordEmbedding = parameter['wordEmbedding']
        del parameter
        train = pickle.load(f)
        sdp_id = train['train_sdp_id']
        labels = train['train_labels']
        del train
        test = pickle.load(f)
        test_sdp_id = test['test_sdp_id']
        test_labels = test['test_labels']
        del test
    test_sdp_id_padding = sequence.pad_sequences(test_sdp_id, maxlen=47, truncating='post', padding='post')

    print('Testing set: ', test_sdp_id_padding.shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.variable_scope("model"):
            eval_model = Model(False, wordEmbedding)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        sess.run(tf.global_variables_initializer())

        # 测试阶段
        model_file = tf.train.latest_checkpoint('./../resource/model_66.8/')
        saver.restore(sess, model_file)
        feed_dic = {eval_model.sdp_ids: test_sdp_id_padding,
                    eval_model.labels: test_labels}
        test_accuracy, test_predictions = sess.run([eval_model.accuracy, eval_model.prob], feed_dict=feed_dic)

        print('---------------------------------------------------------------------------')
        test_precision = precision_each_class(test_predictions, test_labels)
        test_recall = recall_each_class(test_predictions, test_labels)
        test_f1 = f1_each_class_precision_recall(test_precision, test_recall)
        test_count = class_label_count(test_labels)
        print("test accuracy: %.1f%%" % (test_accuracy * 100))
        print_out(test_precision, test_recall, test_f1, test_count)
