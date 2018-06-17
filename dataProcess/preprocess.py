from nltk.parse.stanford import StanfordDependencyParser
import os
import networkx as nx
import xml.dom.minidom
import codecs
from six.moves import cPickle as pickle

os.environ['STANFORD_PARSER'] = './../lib/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = './../lib/stanford-parser-3.9.1-models.jar'


def category(str1):
    if str1[:str1.find('(')] == "USAGE":
        return 0
    elif str1[:str1.find('(')] == "RESULT":
        return 1
    elif str1[:str1.find('(')] == "MODEL-FEATURE":
        return 2
    elif str1[:str1.find('(')] == "PART_WHOLE":
        return 3
    elif str1[:str1.find('(')] == "TOPIC":
        return 4
    elif str1[:str1.find('(')] == "COMPARE":
        return 5


def get_middle_sentence(sentence, entity, tab2_string):
    if entity.nodeType == 3:
        sentence_demo = sentence + entity.data
        return get_middle_sentence(sentence_demo, entity.nextSibling, tab2_string)
    elif entity.nodeName == "entity":
        if entity.getAttribute("id") == tab2_string:
            return sentence
        else:
            sentence_demo = sentence + entity.firstChild.data
            return get_middle_sentence(sentence_demo, entity.nextSibling, tab2_string)
    else:
        return sentence


def get_before_sentence(sentence, entity):
    if entity is not None:
        if entity.nodeType == 3:
            if entity.data.find('.') != -1 or entity.data.find(';') != -1 or \
                            entity.data.find('?') != -1 or entity.data.find('!') != -1:
                sentence_demo = entity.data[entity.data.find('.')+1:] + sentence
                return sentence_demo
            else:
                sentence_demo = entity.data + sentence
                return get_before_sentence(sentence_demo, entity.previousSibling)
        elif entity.nodeName == "entity":
            if entity.firstChild.data.find('.') != -1 or entity.firstChild.data.find(';') != -1 or \
                            entity.firstChild.data.find('?') != -1 or entity.firstChild.data.find('!') != -1:
                sentence_demo = entity.firstChild.data[entity.firstChild.data.find('.')+1:] + sentence
                return sentence_demo
            else:
                sentence_demo = entity.firstChild.data + sentence
                return get_before_sentence(sentence_demo, entity.previousSibling)
        else:
            return sentence
    else:
        return sentence


def get_after_sentence(sentence, entity):
    if entity is not None:
        if entity.nodeType == 3:
            if entity.data.find('.') != -1 or entity.data.find(';') != -1 or \
                            entity.data.find('?') != -1 or entity.data.find('!') != -1:
                sentence_demo = sentence + entity.data[:entity.data.find('.')]
                return sentence_demo
            else:
                sentence_demo = sentence + entity.data
                return get_after_sentence(sentence_demo, entity.nextSibling)
        elif entity.nodeName == "entity":
            if entity.firstChild.data.find('.') != -1 or entity.firstChild.data.find(';') != -1 or \
                            entity.firstChild.data.find('?') != -1 or entity.firstChild.data.find('!') != -1:
                sentence_demo = sentence + entity.firstChild.data[:entity.firstChild.data.find('.')]
                return sentence_demo
            else:
                sentence_demo = sentence + entity.firstChild.data
                return get_after_sentence(sentence_demo, entity.nextSibling)
        else:
            return sentence
    else:
        return sentence


def get_sentences_labels(text_file_path, relation_file_path):
    dom = xml.dom.minidom.parse(text_file_path)
    root = dom.documentElement
    entities = root.getElementsByTagName("entity")

    sentences = []
    labels = []
    entity_id_pair = dict()
    relation_entity = []
    with codecs.open(relation_file_path, 'r', 'utf-8') as f:
        for string_wyd in f.readlines():
            labels.append(category(string_wyd))
            tab1_string = string_wyd[string_wyd.find('(') + 1:string_wyd.find(',')]
            if string_wyd.find('REVERSE') == -1:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(')')]
                relation_entity.append((tab1_string, tab2_string))
            else:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(',', string_wyd.find(',') + 1)]
                relation_entity.append((tab2_string, tab1_string))
            for entity in entities:
                if entity.getAttribute("id") == tab1_string:
                    try:
                        before = get_before_sentence("", entity.previousSibling).translate(
                            str.maketrans('/-\n', '   ')).strip().lower()
                        entity1 = entity.firstChild.data.translate(str.maketrans('/-\n', '   ')).strip().lower()
                        entity_id_pair[tab1_string] = entity1
                        mid = get_middle_sentence("", entity.nextSibling, tab2_string).translate(
                            str.maketrans('/-\n', '   ')).strip().lower()
                    except:
                        print(tab1_string)
                if entity.getAttribute("id") == tab2_string:
                    try:
                        entity2 = entity.firstChild.data.translate(str.maketrans('/-\n', '   ')).strip().lower()
                        entity_id_pair[tab2_string] = entity2
                        after = get_after_sentence("", entity.nextSibling).translate(
                            str.maketrans('/-\n', '   ')).strip().lower()
                        sentences.append(before + ' ' + tab1_string + ' ' + mid + ' ' + tab2_string + ' ' + after)
                        break
                    except:
                        print(tab2_string)
    return sentences, labels, entity_id_pair, relation_entity


def get_sentence_sdp(dependency_trees, relation_entity, entity_id_pair):
    dep_trees = []
    dep_tree = []
    for item in dependency_trees:
        if item == '':
            dep_trees.append(dep_tree)
            dep_tree = []
        else:
            dep_tree.append(item)

    sdps = []
    for i in range(len(dep_trees)):
        edges = []
        for item in dep_trees[i]:
            item_head = item[item.find('(') + 1:item.find(',')]
            item_tail = item[item.find(',') + 2:item.find(')')]
            edges.append((item_head, item_tail))
            if item_head[:item_head.rfind('-')] == relation_entity[i][0]:
                source = item_head
            if item_tail[:item_tail.rfind('-')] == relation_entity[i][0]:
                source = item_tail
            if item_head[:item_head.rfind('-')] == relation_entity[i][1]:
                target = item_head
            if item_tail[:item_tail.rfind('-')] == relation_entity[i][1]:
                target = item_tail
        graph = nx.Graph(edges)
        shortest_path = nx.shortest_path(graph, source=source, target=target)
        sdp = ''
        for index in range(len(shortest_path)-1):
            for item in dep_trees[i]:
                if item.find(shortest_path[index]) != -1 and item.find(shortest_path[index+1]) != -1:
                    if item.find(shortest_path[index]) > item.find(shortest_path[index+1]):
                        sdp = sdp + shortest_path[index][:shortest_path[index].rfind('-')] + ' <<< ' \
                              + item[:item.find('(')] + ' <<< '
                    else:
                        sdp = sdp + shortest_path[index][:shortest_path[index].rfind('-')] + ' >>> ' \
                              + item[:item.find('(')] + ' >>> '
        sdp = sdp + shortest_path[index+1][:shortest_path[index+1].rfind('-')]
        old_head = sdp[:sdp.find(' ')]
        new_head = entity_id_pair[old_head]
        old_tail = sdp[sdp.rfind(' ')+1:]
        new_tail = entity_id_pair[old_tail]
        sdps.append(sdp.replace(old_head, new_head).replace(old_tail, new_tail))
    return sdps


if __name__ == "__main__":
    # test_sentences = get_sentences_labels('./../resource/original/1.2.test.text.xml',
    #                                       './../resource/original/keys.test.1.2.txt')
    #
    # with open('./../resource/generated/sentences_1.2_test.pickle', 'wb') as f:
    #     test = {
    #         'test_sentences': test_sentences[0],
    #         'test_labels': test_sentences[1],
    #         'test_entity_id_pair': test_sentences[2],
    #         'test_relation_entity': test_sentences[3]
    #     }
    #     pickle.dump(test, f, protocol=2)

    with open('./../resource/generated/sentences_1.2_test.pickle', 'rb') as f:
        test = pickle.load(f)
        test_sentences = test['test_sentences']
        test_entity_id_pair = test['test_entity_id_pair']
        test_relation_entity = test['test_relation_entity']
        test_labels = test['test_labels']
        del test

    # dep_parser = StanfordDependencyParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

    # test_dependency_tree = list(dep_parser.raw_parse_sents(test_sentences))

    # with open('./../resource/generated/dependency_tree_1.2_test.pickle', 'wb') as f:
    #     test = {
    #         'test_dependency_tree': test_dependency_tree,
    #         'test_labels': test_labels
    #     }
    #     pickle.dump(test, f, protocol=2)

    with open('./../resource/generated/dependency_tree_1.2_test.pickle', 'rb') as f:
        test = pickle.load(f)
        test_dependency_tree = test['test_dependency_tree']
        del test

    test_sdp = get_sentence_sdp(test_dependency_tree, test_relation_entity, test_entity_id_pair)

    with open('./../resource/generated/sdp_1.2_test.pickle', 'wb') as f:
        test = {
            'test_sdp': test_sdp,
            'test_labels': test_labels
        }
        pickle.dump(test, f, protocol=2)
    print('end')

    # with open('./../resource/generated/sdp.pickle', 'rb') as f:
    #     train = pickle.load(f)
    #     train_sdp = train['train_sdp']
    #     train_labels = train['train_labels']
    #     del train
    #     test = pickle.load(f)
    #     test_sdp = test['test_sdp']
    #     test_labels = test['test_labels']
    #     del test
