import networkx as nx
import spacy
nlp = spacy.load('en')

# https://spacy.io/docs/usage/processing-text
document = nlp("Traditional 1001.5 use a histogram of 1001.7 as the document representation but oral "
               "communication may offer additional indices such as the time and place of the rejoinder and "
               "the attendance.")

print('document: {0}'.format(document))

# Load spacy's dependency tree into a networkx graph
edges = []
for token in document:
    # FYI https://spacy.io/docs/api/token
    for child in token.children:
        edges.append(('{0}-{1}'.format(token.lower_,token.i),
                      '{0}-{1}'.format(child.lower_,child.i)))

graph = nx.Graph(edges)

# https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.shortest_paths.html
print(nx.shortest_path_length(graph, source='1001.5-1', target='1001.7-6'))
print(nx.shortest_path(graph, source='1001.5-1', target='1001.7-6'))
print(nx.shortest_path(graph, source='1001.7-6', target='1001.5-1'))