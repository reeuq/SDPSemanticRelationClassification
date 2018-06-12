from __future__ import print_function
import numpy as np


n_class = 6


def print_out(precision, recall, f1, count):
    print("%24s%10s%10s%10s%10s"%("","precision","recall","f1","count"))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("usage", precision[0], recall[0], f1[0], count[0]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("result", precision[1], recall[1], f1[1], count[1]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("model_feature", precision[2], recall[2], f1[2], count[2]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("part_whole", precision[3], recall[3], f1[3], count[3]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("topic", precision[4], recall[4], f1[4], count[4]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("compare", precision[5], recall[5], f1[5], count[5]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("average/sum", sum(precision)/len(precision), sum(recall)/len(recall),
                                          sum(f1)/len(f1), sum(count)))


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def class_label_count(labels):
    categories = [-1] * n_class
    for a in range(0, n_class):
        categories[a] = np.sum(np.argmax(labels, 1) == a)
    return categories


def precision_each_class(predictions, labels):
    categories = [-1] * n_class
    for a in range(0, n_class):
        zeros = np.zeros_like(labels)
        zeros[:, a] = 1
        if np.sum(np.argmax(predictions, 1) == np.argmax(zeros, 1)) == 0:
            categories[a] = 0
        else:
            categories[a] = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                            & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                             / np.sum(np.argmax(predictions, 1) == np.argmax(zeros, 1)))
    return categories


def recall_each_class(predictions, labels):
    categories = [-1] * n_class
    for a in range(0, n_class):
        zeros = np.zeros_like(labels)
        zeros[:, a] = 1
        if np.sum(np.argmax(labels, 1) == np.argmax(zeros, 1)) == 0:
            categories[a] = 0
        else:
            categories[a] = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                            & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                             / np.sum(np.argmax(labels, 1) == np.argmax(zeros, 1)))
    return categories


def f1_each_class(predictions, labels):
    categories = [-1] * n_class
    for a in range(0, n_class):
        zeros = np.zeros_like(labels)
        zeros[:, a] = 1
        if np.sum(np.argmax(predictions, 1) == np.argmax(zeros, 1)) == 0:
            precision = 0
        else:
            precision = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                        & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                         / np.sum(np.argmax(predictions, 1) == np.argmax(zeros, 1)))
        if np.sum(np.argmax(labels, 1) == np.argmax(zeros, 1)) == 0:
            recall = 0
        else:
            recall = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                     & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                      / np.sum(np.argmax(labels, 1) == np.argmax(zeros, 1)))
        if precision + recall == 0:
            categories[a] = 0
        else:
            categories[a] = (2 * precision * recall) / (precision + recall)
    return categories


def f1_each_class_precision_recall(precision, recall):
    categories = [-1] * n_class
    for a in range(0, n_class):
        if precision[a] + recall[a] == 0:
            categories[a] = 0
        else:
            categories[a] = (2 * precision[a] * recall[a]) / (precision[a] + recall[a])
    return categories
