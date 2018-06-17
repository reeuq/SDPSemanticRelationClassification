from six.moves import cPickle as pickle
from dataProcess import make_dict
import numpy as np

if __name__ == '__main__':
    with open('./../resource/generated/input.pickle', 'rb') as f:
        parameter = pickle.load(f)
        dictionary = parameter['dictionary']
        del parameter

    with open('./../resource/generated/sdp_1.2_test.pickle', 'rb') as f:
        test = pickle.load(f)
        test_sdp = test['test_sdp']
        test_labels = test['test_labels']
        del test

    test_sdp_id, test_max_len = make_dict.get_sentence_id(test_sdp, dictionary)
    test_labels = make_dict.reformat(np.array(test_labels))

    with open('./../resource/generated/input_1.2_test.pickle', 'wb') as f:
        test = {
            'test_sdp_id': test_sdp_id,
            'test_labels': test_labels,
        }
        pickle.dump(test, f, protocol=2)
    print("test max len: ", test_max_len)
