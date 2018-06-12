from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser
from nltk.tokenize.stanford import StanfordTokenizer
import os

os.environ['STANFORD_PARSER'] = './../lib/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = './../lib/stanford-parser-3.9.1-models.jar'

# stanford_tokenizer = StanfordTokenizer(path_to_jar='./../lib/stanford-parser.jar')
#
# print(stanford_tokenizer.tokenize("My dog also likes eating sausage.\r\n My dog also likes eating sausage."))
# print("token end")


dep_parser = StanfordDependencyParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
# parser = StanfordParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

# print([parse.tree() for parse in dep_parser.raw_parse("My dog also likes eating sausage.")])
# print([list(parse.triples()) for parse in dep_parser.raw_parse("My P05-1067.1 also likes P05-1067.2 sausage.")])
# print('middle')
# print([list(parse.triples()) for parse in dep_parser.raw_parse("My dog also likes eating sausage.")])

# for sentence in parser.raw_parse("My dog also likes eating sausage."):
#     sentence.draw()
sentences = ["Traditional H01-1001.5 use a histogram of H01-1001.7 as the document representation but oral ",
             "communication may offer additional indices such as the time and place of the rejoinder and ",
             "the attendance."]
a = dep_parser.raw_parse_sents(sentences)
b=list(a)
for raw in a:
    print(raw)
